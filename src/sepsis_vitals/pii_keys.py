"""
sepsis_vitals.pii_keys — the PII keyring: which key encrypts, which keys may decrypt.

Configuration (environment)
---------------------------
``SEPSIS_PII_KEY``
    The current key, base64 of 32 random bytes. New values are encrypted and
    new blind indexes computed with it.
``SEPSIS_PII_KEY_ID``
    Identifier of the current key (``[a-z0-9][a-z0-9_-]{0,23}``). Unset means
    ``legacy``: the key in use before key IDs existed.
``SEPSIS_PII_PREVIOUS_KEYS``
    Comma-separated ``id:base64key`` pairs that may still decrypt and match
    blind indexes during a rotation, but never encrypt. Example:
    ``legacy:<old key>``.

Key IDs and formats
-------------------
* ``legacy`` is the reserved ID for the pre-rotation key. It keeps the
  original behaviour, so existing data stays readable: ciphertext
  ``enc:<b64(nonce|ct|tag)>`` with AES-256-GCM under the raw key, and blind
  indexes as HMAC-SHA256 under the raw key.
* Any other ID writes ``enc:v2:<id>:<b64(nonce|ct|tag)>``. The key ID is bound
  to the ciphertext as associated data, so relabelling fails authentication.
  Encryption and blind-index subkeys are derived separately with HKDF-SHA256.

Rotation is described in docs/pii_key_rotation.md. Rotating keys protects
data written from now on and re-protects stored data. It does **not** undo
an exposure: anyone who obtained an old key can still read every copy
(backups, snapshots, exports) encrypted under it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LEGACY_ID = "legacy"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_V2_PREFIX = "enc:v2:"


class KeyringError(ValueError):
    """Invalid key configuration. Messages never contain key material."""


@dataclass(frozen=True)
class PIIKey:
    kid: str
    secret: bytes = field(repr=False)

    @property
    def legacy(self) -> bool:
        return self.kid == LEGACY_ID

    def _derive(self, purpose: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"sepsis-vitals/" + purpose + b"/" + self.kid.encode()).derive(self.secret)

    @property
    def encryption_key(self) -> bytes:
        return self.secret if self.legacy else self._derive(b"pii-encryption")

    @property
    def index_key(self) -> bytes:
        return self.secret if self.legacy else self._derive(b"blind-index")

    def aad(self) -> Optional[bytes]:
        return None if self.legacy else b"sepsis-vitals:pii:v2:" + self.kid.encode()


@dataclass(frozen=True)
class Keyring:
    current: PIIKey
    previous: Tuple[PIIKey, ...] = ()

    @property
    def keys(self) -> Tuple[PIIKey, ...]:
        return (self.current,) + self.previous

    def get(self, kid: str) -> Optional[PIIKey]:
        return next((k for k in self.keys if k.kid == kid), None)

    # -- encryption ----------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = self.current
        nonce = os.urandom(12)
        ct = AESGCM(key.encryption_key).encrypt(nonce, plaintext.encode("utf-8"), key.aad())
        payload = base64.b64encode(nonce + ct).decode("ascii")
        return f"enc:{payload}" if key.legacy else f"{_V2_PREFIX}{key.kid}:{payload}"

    def decrypt(self, token: str) -> str:
        """Decrypt *token*; raises ValueError naming only the format or key ID."""
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        kid, raw = parse_token(token)
        if kid is None:
            candidates = list(self.keys)  # legacy format: the AEAD tag identifies the key
        else:
            key = self.get(kid)
            if key is None:
                raise ValueError(f"key id '{kid}' is not configured")
            candidates = [key]
        nonce, ciphertext = raw[:12], raw[12:]
        for key in candidates:
            aad = None if kid is None else key.aad()
            secret = key.secret if kid is None else key.encryption_key
            try:
                return AESGCM(secret).decrypt(nonce, ciphertext, aad).decode("utf-8")
            except InvalidTag:
                continue
        raise ValueError("no configured key decrypts this value" if kid is None
                         else f"authentication failed under key id '{kid}'")

    def key_id_of(self, token: str) -> Optional[str]:
        """Key ID that decrypts *token* (``legacy``-format values are tried)."""
        kid, _ = parse_token(token)
        keys = [k for k in self.keys if kid is None or k.kid == kid]
        for key in keys:
            try:
                Keyring(key).decrypt(token)
                return key.kid
            except ValueError:
                continue
        return None

    # -- blind indexes ---------------------------------------------------------------

    def blind_index(self, value: str, key: Optional[PIIKey] = None) -> str:
        key = key or self.current
        return hmac.new(key.index_key, value.lower().encode("utf-8"), hashlib.sha256).hexdigest()

    def blind_index_candidates(self, value: str) -> List[str]:
        """Index of *value* under every configured key, current first."""
        return [self.blind_index(value, k) for k in self.keys]


def parse_token(token: str) -> Tuple[Optional[str], bytes]:
    """(key id or None for the legacy format, nonce|ct|tag). Raises ValueError."""
    if not token.startswith("enc:"):
        raise ValueError("value is not encrypted")
    if token.startswith(_V2_PREFIX):
        kid, sep, payload = token[len(_V2_PREFIX):].partition(":")
        if not sep or not _ID_PATTERN.match(kid):
            raise ValueError("malformed versioned ciphertext")
    else:
        kid, payload = None, token[4:]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        raise ValueError("ciphertext is not valid base64") from None
    if len(raw) < 12 + 16:
        raise ValueError("ciphertext is truncated")
    return kid, raw


def _decode_key(kid: str, raw: str) -> PIIKey:
    if not _ID_PATTERN.match(kid):
        raise KeyringError(f"invalid key id '{kid}' (use lowercase letters, digits, '-' or '_', at most 24)")
    try:
        secret = base64.b64decode(raw.strip(), validate=True)
    except (ValueError, TypeError):
        raise KeyringError(f"key '{kid}' is not valid base64") from None
    if len(secret) != 32:
        raise KeyringError(f"key '{kid}' must decode to 32 bytes, got {len(secret)}")
    return PIIKey(kid, secret)


def keyring_from_env(env: Optional[Dict[str, str]] = None) -> Optional[Keyring]:
    """The configured keyring, or None when no current key is set (development)."""
    env = dict(os.environ if env is None else env)
    raw = env.get("SEPSIS_PII_KEY", "")
    if not raw or raw == "REPLACE_ME_BASE64_32_BYTES":
        if env.get("SEPSIS_PII_PREVIOUS_KEYS"):
            raise KeyringError("SEPSIS_PII_PREVIOUS_KEYS is set but SEPSIS_PII_KEY is not")
        return None
    current = _decode_key(env.get("SEPSIS_PII_KEY_ID") or LEGACY_ID, raw)
    previous: List[PIIKey] = []
    for item in filter(None, (p.strip() for p in env.get("SEPSIS_PII_PREVIOUS_KEYS", "").split(","))):
        kid, sep, value = item.partition(":")
        if not sep:
            raise KeyringError("SEPSIS_PII_PREVIOUS_KEYS entries must be 'id:base64key'")
        previous.append(_decode_key(kid.strip(), value))
    seen = {current.kid}
    for key in previous:
        if key.kid in seen:
            raise KeyringError(f"key id '{key.kid}' is configured more than once")
        seen.add(key.kid)
        if hmac.compare_digest(key.secret, current.secret):
            raise KeyringError(f"previous key '{key.kid}' is the same key as the current key")
    return Keyring(current, tuple(previous))

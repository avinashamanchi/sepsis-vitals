"""
tests/test_site_setup.py — site installation helpers: the secrets file is
complete, private and never overwritten; operator account creation enforces
the password policy and never takes the password as an argument.
"""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_site", ROOT / "scripts" / "setup_site.py")
setup_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_site)

REQUIRED = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "SEPSIS_PII_KEY", "SEPSIS_CONFIG_KEY",
            "SEPSIS_WEBHOOK_SECRET", "SEPSIS_TOKEN_SECRET", "JWT_PRIVATE_KEY", "JWT_PUBLIC_KEY")


def _parse(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line and not line.startswith("#"))


def test_env_file_is_complete_private_and_unique(tmp_path, capsys):
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    assert setup_site.main(["--env-file", str(a)]) == 0
    setup_site.main(["--env-file", str(b)])
    out = capsys.readouterr().out
    env_a, env_b = _parse(a), _parse(b)
    for key in REQUIRED:
        assert env_a[key] and env_a[key] != env_b[key], key
        assert env_a[key] not in out  # nothing secret is printed
    assert stat.S_IMODE(a.stat().st_mode) == 0o600
    assert env_a["SEPSIS_ENV"] == "production"
    assert env_a["JWT_PRIVATE_KEY"].startswith("-----BEGIN RSA PRIVATE KEY-----\\n")
    import base64
    assert len(base64.b64decode(env_a["SEPSIS_PII_KEY"])) == 32


def test_existing_env_file_is_never_overwritten(tmp_path):
    path = tmp_path / ".env"
    path.write_text("SEPSIS_PII_KEY=existing\n")
    assert setup_site.main(["--env-file", str(path)]) == 1
    assert path.read_text() == "SEPSIS_PII_KEY=existing\n"


def test_generated_keys_work_with_the_application(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    setup_site.main(["--env-file", str(path)])
    env = _parse(path)
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import serialization

    Fernet(env["SEPSIS_CONFIG_KEY"].encode())
    serialization.load_pem_private_key(env["JWT_PRIVATE_KEY"].replace("\\n", "\n").encode(), password=None)
    serialization.load_pem_public_key(env["JWT_PUBLIC_KEY"].replace("\\n", "\n").encode())


def test_create_user_enforces_policy_and_allows_the_first_admin():
    from sepsis_vitals.auth.admin_cli import create_user
    from sepsis_vitals.auth.service import AuthServiceError, DuplicateEmailError
    from sepsis_vitals.db import SessionLocal, User, init_db

    init_db()
    db = SessionLocal()
    try:
        admin = create_user(db, "first.admin@site.example", "Correct-Horse-Battery-9!", "system_admin", None)
        assert admin.role == "system_admin" and admin.password_hash != "Correct-Horse-Battery-9!"
        with pytest.raises(DuplicateEmailError):
            create_user(db, "first.admin@site.example", "Correct-Horse-Battery-9!", "system_admin", None)
        with pytest.raises(AuthServiceError):
            create_user(db, "weak@site.example", "short", "nurse", "WARD-A")
        with pytest.raises(ValueError, match="--site"):
            create_user(db, "nosite@site.example", "Correct-Horse-Battery-9!", "nurse", None)
        assert db.query(User).filter(User.role == "system_admin").count() >= 1
    finally:
        db.close()


def test_create_user_cli_has_no_password_argument():
    from sepsis_vitals.auth import admin_cli

    with pytest.raises(SystemExit):
        admin_cli.main(["create-user", "--email", "x@site.example", "--role", "nurse", "--password", "p"])

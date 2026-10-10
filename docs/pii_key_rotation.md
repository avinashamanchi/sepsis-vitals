# PII key rotation

This is the procedure for replacing the key that protects personal data at
rest. It covers rotation and compromise; the two are handled differently.

Nothing in this repository generates, installs or retires production keys.
Each step below is an operator action, and the "Approvals" section lists the
steps that need sign-off.

## What the key protects

| Data | Where | Protection | Re-keyed by the tool |
|---|---|---|---|
| `users.email` | database | AES-256-GCM ciphertext | yes |
| `users.email_hash` | database | HMAC-SHA256 blind index of the email | yes, recomputed from the decrypted email |
| `users.totp_secret` | database | AES-256-GCM ciphertext | yes |
| `users.mfa_recovery_hashes` | database | HMAC of each unused recovery code | **no**: the codes are not stored |
| `patients.external_id` (MRN) | database | AES-256-GCM ciphertext | yes |
| `patients.external_id_hash` | database | HMAC blind index of the MRN | yes, recomputed from the decrypted MRN |
| `tracked_alerts.patient_id`, `alert_audit_trail.user_id`/`detail` | `alert_escalation.v1.db` | AES-256-GCM ciphertext | yes (`--escalation-store`) |
| prediction history references | `patient_state.v1.db` | HMAC of the patient id | **no**: moved to the current key the next time the patient is scored |
| `ref:...` correlation ids | log lines | HMAC prefix | no (log correlation across a rotation is lost) |
| database backups, snapshots, exports | outside the application | whatever key was current when they were taken | no |

Blind indexes use the same key as encryption: the raw key for `legacy`, and
an HKDF-derived subkey for versioned keys. Rotating the key therefore also
rotates the indexes.

## Configuration

| Variable | Meaning |
|---|---|
| `SEPSIS_PII_KEY` | Current key: base64 of 32 random bytes. Encrypts all new values. |
| `SEPSIS_PII_KEY_ID` | ID of the current key. Unset means `legacy` (the key in use before key IDs existed). |
| `SEPSIS_PII_PREVIOUS_KEYS` | `id:base64key,...`: keys that may still decrypt and match lookups, but never encrypt. |

**The pre-rotation key must keep the ID `legacy`.** Existing ciphertexts
(`enc:...`) and blind indexes were made with it using the original scheme.
Under any other ID, lookups by email or MRN stop matching. `verify` reports
this as blind indexes that "match no key".

Generate a key on a trusted host and put it straight into the secret store.
Do not commit it, paste it into tickets, or store it next to the data it
protects:

```bash
python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
```

## Procedure

Every application instance must run a keyring-aware build (this release or
later) before step 2. Older builds cannot read `enc:v2:` values.

1. **Inventory (read-only).** Count values per format and key, and blind
   indexes per state, with no values printed:
   `python -m sepsis_vitals.pii_rotation inventory --escalation-store /app/state/alert_escalation.v1.db`
   Any `plaintext` value (written while no key was configured) or
   `unreadable` value must be understood before continuing.
2. **Add the new key as readable.** On every instance, set
   `SEPSIS_PII_PREVIOUS_KEYS=k2026a:<new key>` and leave the current key
   unchanged. Roll this out completely before step 3. Afterwards any
   instance can read values written under either key.
3. **Promote the new key.** On every instance, set `SEPSIS_PII_KEY=<new key>`,
   `SEPSIS_PII_KEY_ID=k2026a` and `SEPSIS_PII_PREVIOUS_KEYS=legacy:<old key>`.
   New values are written as `enc:v2:k2026a:...`. Lookups match indexes
   under both keys, so logins, MRN lookups, recovery codes and prediction
   history keep working, and no duplicate patient or user can be created
   under the new index.
4. **Dry run.** `python -m sepsis_vitals.pii_rotation rotate --escalation-store ...`
   This reports what would be re-keyed and writes nothing.
5. **Re-key (needs approval: it rewrites stored personal data).**
   `python -m sepsis_vitals.pii_rotation rotate --apply --batch-size 500 [--max-batches N] --escalation-store ...`
   - Each batch is one transaction. On PostgreSQL its rows are locked
     (`FOR UPDATE`), and a row is written only if it still holds the value
     that was read.
   - If the command is interrupted, it loses at most the current batch.
     Run it again to continue.
   - Rows no configured key can decrypt stop the run (the batch is rolled
     back) unless `--skip-unreadable` is given; they are listed by `ref:`
     only.
   - `--encrypt-plaintext` also encrypts values stored without encryption.
6. **Verify.** `python -m sepsis_vitals.pii_rotation verify --escalation-store ...`
   exits 0 only when every value and blind index is under the current key.

## Retiring the old key

Remove the old key from `SEPSIS_PII_PREVIOUS_KEYS` only when all of these
hold:

- `verify` exits 0 on the production database and every escalation store.
- **Recovery codes:** `verify` and `inventory` report
  `users_with_recovery_codes`. Recovery codes issued before step 3 stop
  working once the old key is removed. Ask those users to regenerate their
  codes (disable and re-enable MFA, or an administrator `reset-mfa`), or
  accept that the old codes become invalid.
- **Prediction history** for patients not scored since step 3 is no longer
  found (it ages out under `cleanup_old_records`). Trends and baselines for
  those patients start again.
- **Backups:** every backup, snapshot or export taken before the re-key is
  encrypted under the old key. Keep the old key, escrowed offline and
  access-controlled, until the last such backup has passed its retention
  period. Destroying the key earlier makes those backups unreadable.

Retiring a key cannot be undone for data that still needs it. Treat the
escrow copy as the rollback.

## Rotation is not compromise recovery

Rotation re-protects stored data and protects data written from now on. It
does **not** undo an exposure. Anyone who obtained the old key can still:

- decrypt every copy of data encrypted under it: database backups,
  snapshots, replicas, exports and any copied tables;
- test guessed emails or MRNs against blind indexes computed with it.

If a key may have been exposed, treat it as a security incident:

1. Contain it and establish scope (which data, copies and period).
2. Rotate with the procedure above.
3. Decide, with your privacy officer, which backups to destroy or treat as
   exposed.
4. Assess breach-notification obligations (for example HIPAA or GDPR).
5. Revoke sessions and other credentials that may have been exposed with it.

## Approvals

| Step | Approval needed |
|---|---|
| Generating or installing a production key | Security owner |
| Steps 2-3 (configuration rollout) | Deployment owner |
| Step 5 `--apply` (rewrites personal data) | Data owner, after a backup |
| Retiring the old key and destroying escrow copies | Security owner and data owner |

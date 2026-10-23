# TOTP replay protection

Each TOTP code is accepted at most once, on every path that checks a code:
enrollment confirmation, sign-in and self-service disable. Recovery codes
are single-use, including under concurrent requests. This closes N52
(RFC 6238 section 5.2).

## Policy

- **Time step:** 30 seconds (`auth.jwt.TOTP_INTERVAL_S`).
- **Accepted drift:** the server's current step and one step on either side
  (`TOTP_DRIFT_STEPS = 1`). `auth.jwt.match_totp_step` returns the step the
  code belongs to, trying the newest step first. Input is accepted only as
  six ASCII digits; anything else matches nothing.
- **Monotonic consumption:** `users.totp_last_step` holds the last accepted
  step for the current secret. A code is accepted only if its step is later
  than that. As a result:
  - the same code is refused the second time, on any worker, after any
    restart;
  - a code for an earlier step is refused once a later step has been used.
    After a code from a fast clock (the next step) is accepted, the user
    waits until the server reaches a later step. This is the cost of
    tolerating drift without allowing replays;
  - the code that confirms enrollment is consumed, so the first sign-in
    needs the next code (at most 30 s later).
- **Atomic decision:** one conditional `UPDATE users SET totp_last_step = :step
  WHERE id = :id AND (totp_last_step IS NULL OR totp_last_step < :step)`.
  Of concurrent requests with the same code, in any number of workers, at
  most one matches the row. Recovery codes use the same compare-and-set on
  the stored hash list.
- **Same transaction as the sign-in:** the update is not committed by the
  check. It commits with the successful login, confirmation or disable, and
  rolls back with it. An attempt that fails afterwards does not burn the
  code.
- **Fails closed:** a database error during the check propagates. The
  request fails and no session is issued.
- **Per secret:** starting enrollment (a new secret), disabling MFA and
  administrator reset all clear `totp_last_step`. A new secret starts with
  no history; codes from the old secret no longer verify.
- **What is not consumed:** wrong, malformed or non-matching codes change
  nothing (they still count towards the login lockout).
- **Errors:** a replayed code gets exactly the same response as a wrong
  one. Logs contain no codes, secrets or recovery codes (tested).

## Migration and deployment

Migration `005_totp_replay_protection` (revision `e5f6a7b8c9d0`, after
`d4e5f6a7b8c9`) adds a nullable `users.totp_last_step BIGINT`:

- Existing users get NULL, meaning no step used yet, so their next valid
  code works. No data is rewritten.
- It is additive. The running (old) application ignores the column, so the
  migration can run before the new code is deployed.

Order:

1. Run `alembic upgrade head`. The compose entrypoint does this; on ECS, use
   the one-off run-task in PROJECT_REVIEW.md §9.4 item 8.
2. Deploy the new application version.
3. **Limitation:** while any replica of the previous version still serves
   sign-ins, a code accepted by a new replica can be replayed against an old
   one, which does not check or record steps. Replay protection is complete
   once no old replica remains. Use a stop-then-start deploy, or drain old
   tasks fully, if that window matters.

Rollback: deploying the previous version reverts to "no replay protection".
The column can stay. `alembic downgrade d4e5f6a7b8c9` removes it.

## Tests

- `tests/test_mfa_replay.py` covers:
  - first use, immediate replay, and drift order;
  - malformed and wrong codes, and users without MFA;
  - the enrollment code, disable, reset and re-enrollment;
  - recovery codes, 6 concurrent threads, and 4 worker processes;
  - restart, rollback, database failure and the migration;
  - log redaction and the default policy.
- `tests/test_mfa.py` uses the same controlled clock.
- The PostgreSQL CI job runs both files.
- The compose E2E signs in through the browser with a code and checks that
  the same code is refused when replayed against the 4-worker API.

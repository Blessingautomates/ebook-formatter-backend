"""Credit metering against the Supabase ledger.

The ledger lives in Postgres (`credit_ledger`), and the browser already reads it
directly — the dashboard's balance is a real sum, not a figure this service
invented. What the browser cannot do is *spend*: `credit_ledger` has select-own
and no write policy, because an account that can write its own balance has no
reason to buy credits.

So spending is the backend's job, and this module is the only thing that does
it. Three properties matter, and each is enforced in the database rather than
here:

* **Only the server may debit.** `debit_credits` is a `security definer`
  function whose execute privilege is revoked from `anon` and `authenticated`
  and granted to `service_role`. A client calling it gets a permission error.
* **A debit cannot overdraw.** The balance is read and the row written in one
  transaction, under a per-account advisory lock, so two analyses that start at
  the same moment cannot both pass a check they were supposed to share.
* **A grant cannot be forged.** `ensure_credit_grant` issues the monthly
  allowance once per calendar month, keyed on a unique period, so calling it
  twice is the same as calling it once.

This module adds no arithmetic of its own. It forwards, and it maps Postgres
errors onto answers the API can return.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services.auth import AuthedUser

#: Postgres SQLSTATEs this module recognises, and what they mean here.
UNDEFINED_FUNCTION = "42883"  # the schema has not been applied
INSUFFICIENT_CREDITS = "P0001"  # raised by debit_credits


class CreditsError(Exception):
    """The ledger could not be read or written."""


class InsufficientCredits(CreditsError):
    """The account does not have enough credits for this operation. A 402.

    Carries the numbers so the response can say what was needed and what is
    left, which is the difference between an error and an explanation.
    """

    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"This needs {required:,} credits and the account has {available:,}."
        )


class CreditsUnavailableError(CreditsError):
    """Metering is not configured on this deployment. A 503.

    Raised rather than silently allowing the work through. Free analysis on a
    deployment that meant to charge for it is a bug that only shows up on the
    invoice.
    """


@dataclass(frozen=True)
class CreditPosition:
    """An account's credit standing, as `credit_summary()` reports it.

    Only these three columns exist. The monthly allowance is deliberately *not*
    one of them: it is a property of the plan, and the plan lives in
    `subscriptions`, so the UI derives it from there rather than having the
    database restate it in two places that could disagree.
    """

    balance: int
    spent: int
    period_start: str | None = None


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _url() -> str:
    base = _env("SUPABASE_URL")
    if not base:
        raise CreditsUnavailableError(
            "Credit metering needs SUPABASE_URL to reach the ledger."
        )
    return base.rstrip("/")


def _service_key() -> str:
    key = _env("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise CreditsUnavailableError(
            "Credit metering needs SUPABASE_SERVICE_ROLE_KEY to write to the ledger."
        )
    return key


def credits_configured() -> bool:
    """Whether this deployment can read and spend credits."""
    return bool(_env("SUPABASE_URL") and _env("SUPABASE_SERVICE_ROLE_KEY"))


def _require_httpx() -> Any:
    """httpx, imported lazily so a deployment without metering needs no HTTP client."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise CreditsUnavailableError(
            "Credit metering needs httpx. Add it to requirements.txt and rebuild."
        ) from exc
    return httpx


def _plain_error(payload: Any, status: int) -> CreditsError:
    """Turn a PostgREST error body into something worth reading.

    PostgREST answers with `{"message": ..., "code": ..., "details": ...}`, and
    the code is what says whether the schema is missing or the function raised.
    """
    code = payload.get("code") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
    text = message if isinstance(message, str) and message.strip() else None

    if code == UNDEFINED_FUNCTION:
        return CreditsUnavailableError(
            "The credit functions are missing from the database. "
            "Apply supabase/schema.sql."
        )
    if status == 401 or status == 403:
        return CreditsUnavailableError(
            "The ledger refused the service key. Check SUPABASE_SERVICE_ROLE_KEY."
        )
    # The function raises the literal 'insufficient_credits' as its message, so
    # it survives PostgREST's error envelope. Parsed below, where the amounts
    # are known.
    return CreditsError(text or f"The ledger returned {status}.")


async def _rpc(
    name: str,
    payload: dict[str, Any],
    *,
    token: str,
) -> Any:
    """Call a PostgREST RPC **as the signed-in user**.

    The caller's own token is forwarded so row-level security applies exactly as
    it would in the browser. The backend does not get to see rows the account
    could not read itself.
    """
    httpx = _require_httpx()
    url = f"{_url()}/rest/v1/rpc/{name}"
    headers = {
        "apikey": _env("SUPABASE_ANON_KEY") or token,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 503, see below
        raise CreditsUnavailableError(f"The ledger could not be reached: {exc}") from exc

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON error body is still an error
            body = {}
        raise _plain_error(body, response.status_code)

    try:
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise CreditsError("The ledger returned a response that is not JSON.") from exc


async def ensure_grant(user: AuthedUser) -> None:
    """Issue this month's allowance if it has not been issued yet.

    Idempotent by construction — the function keys on `(user_id, period_start)`
    with a unique constraint and returns early when the row exists — so calling
    it before every metered operation is correct rather than merely harmless.
    """
    await _rpc("ensure_credit_grant", {}, token=user.token)


async def position(user: AuthedUser) -> CreditPosition:
    """Read the account's balance, allowance and spend for the period."""
    rows = await _rpc("credit_summary", {}, token=user.token)
    row = rows[0] if isinstance(rows, list) and rows else rows
    if not isinstance(row, dict):
        raise CreditsError("The ledger did not return a credit summary.")

    def number(key: str) -> int:
        # Postgres bigints arrive through PostgREST as strings, so an int() that
        # trusted the JSON type would raise on every real response.
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    period = row.get("period_start")
    return CreditPosition(
        balance=number("balance"),
        spent=number("spent"),
        period_start=period if isinstance(period, str) else None,
    )


async def debit(
    user: AuthedUser,
    amount: int,
    reason: str,
    ref: str | None = None,
) -> int:
    """Spend `amount` credits and return the new balance.

    Written with the service role, because the ledger has no client write policy
    by design. The check and the write happen together inside `debit_credits`,
    so this cannot overdraw even if two requests arrive at once.
    """
    if amount <= 0:
        raise CreditsError("A debit must be a positive number of credits.")

    httpx = _require_httpx()
    key = _service_key()
    url = f"{_url()}/rest/v1/rpc/debit_credits"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "p_user_id": user.id,
        "p_amount": amount,
        "p_reason": reason,
        "p_ref": ref,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001
        raise CreditsUnavailableError(f"The ledger could not be reached: {exc}") from exc

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = {}

        message = body.get("message") if isinstance(body, dict) else None
        if isinstance(message, str) and "insufficient_credits" in message:
            # The function refuses without saying the numbers, so they are read
            # back to build a message the caller can act on.
            available = 0
            try:
                available = (await position(user)).balance
            except CreditsError:
                # Reporting "0 left" when the balance could not be read would be
                # a guess presented as a fact.
                raise InsufficientCredits(amount, 0) from None
            raise InsufficientCredits(amount, available)

        raise _plain_error(body, response.status_code)

    try:
        result = response.json()
    except Exception as exc:  # noqa: BLE001
        raise CreditsError("The ledger returned a response that is not JSON.") from exc

    # `returns integer`, so the body is a bare number; PostgREST may still wrap
    # it in a list depending on the function's shape.
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        result = result.get("debit_credits", result.get("balance"))

    try:
        return int(result)
    except (TypeError, ValueError) as exc:
        raise CreditsError("The ledger did not return a balance after the debit.") from exc

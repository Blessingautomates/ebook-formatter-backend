"""Optional Supabase authentication for the compute routes.

The backend was unauthenticated: it took a file, returned measurements, and had
no idea who was asking. That is fine for a stateless formatter, but not for one
that charges for its work — a credit ledger needs a subject to charge.

So this module is deliberately **optional and additive**:

* With nothing configured, `current_user` returns `None` and every route behaves
  exactly as it did before. A self-hosted deployment that never sets a Supabase
  secret loses nothing.
* With a secret configured, a route that asks for the caller gets a verified
  `AuthedUser` or a 401. It never falls back to anonymous when it has been told
  who to expect — an authentication system that quietly stops authenticating is
  worse than none, because the ledger would silently credit the wrong subject.

Two things are checked, not one: the signature, and the claims that make the
token usable here. A token that verifies but has expired, or that was issued for
a different audience, is not a session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

#: The audience Supabase issues for a signed-in user of this project.
SUPABASE_AUDIENCE = "authenticated"


class AuthError(Exception):
    """The caller's token is missing, malformed, expired or not ours. A 401."""


class AuthUnavailableError(Exception):
    """Authentication is configured but cannot be performed. A 503, not a 401.

    Separate from `AuthError` on purpose: "I cannot check who you are" and "you
    are not who you say you are" deserve different answers, and only one of them
    is the caller's fault.
    """


@dataclass(frozen=True)
class AuthedUser:
    """A verified caller.

    `token` is kept because Supabase is the data of record and its row-level
    security is what protects a row. Passing the user's own token through means
    the database enforces the same rules for a backend call as it does for a
    browser call, rather than the backend asserting them.
    """

    id: str
    email: str | None
    token: str


def _secret() -> str | None:
    """The HS256 shared secret, if this project still uses one."""
    value = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    return value or None


def _jwks_url() -> str | None:
    """The JWKS endpoint, for a project on asymmetric signing keys."""
    explicit = os.environ.get("SUPABASE_JWKS_URL", "").strip()
    if explicit:
        return explicit

    base = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    return f"{base}/auth/v1/.well-known/jwks.json" if base else None


def auth_configured() -> bool:
    """Whether this deployment can verify a token at all."""
    return _secret() is not None or _jwks_url() is not None


#: Cached so the JWKS document is fetched once per process rather than once per
#: request. Supabase rotates keys rarely; a restart picks up a rotation.
_jwks_client: Any = None


def _decode(token: str) -> dict[str, Any]:
    """Verify a token and return its claims.

    PyJWT is imported lazily, the same way the exporter imports its renderers:
    a deployment that never enables auth should not have to install it, and one
    that enables auth without installing it gets a clear message rather than an
    ImportError traceback.
    """
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise AuthUnavailableError(
            "Authentication is configured but PyJWT is not installed. "
            "Add PyJWT[crypto] to requirements.txt and rebuild."
        ) from exc

    secret = _secret()
    try:
        if secret is not None:
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience=SUPABASE_AUDIENCE,
            )

        # Asymmetric keys. `PyJWKClient` caches the document and re-fetches it
        # when it meets a key id it does not know, which is what makes a
        # rotation a non-event.
        global _jwks_client
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(_jwks_url())
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=SUPABASE_AUDIENCE,
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Your session has expired. Sign in again.") from exc
    except jwt.InvalidAudienceError as exc:
        raise AuthError("That token was not issued for this application.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("That session token is not valid.") from exc


def current_user(authorization: str | None) -> AuthedUser | None:
    """The verified caller, `None` when auth is switched off, or raise.

    Raises `AuthError` when auth is configured and the caller did not present a
    usable token. It deliberately does not return `None` in that case: a route
    that went on to charge an anonymous subject would be charging nobody.
    """
    if not auth_configured():
        return None

    if not authorization or not authorization.strip():
        raise AuthError("Sign in to use this endpoint.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Send the session token as `Authorization: Bearer <token>`.")

    token = token.strip()
    claims = _decode(token)

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        # A token with no subject cannot be charged to anyone.
        raise AuthError("That session token does not identify an account.")

    email = claims.get("email")
    return AuthedUser(
        id=subject,
        email=email if isinstance(email, str) else None,
        token=token,
    )

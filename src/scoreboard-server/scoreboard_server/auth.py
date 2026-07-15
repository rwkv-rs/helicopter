from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .settings import AuthPrincipal


_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Authenticator:
    tokens: tuple[tuple[str, AuthPrincipal], ...]

    def authenticate(self, authorization: str | None) -> AuthPrincipal:
        scheme, separator, supplied = (authorization or "").partition(" ")
        if not separator or scheme.lower() != "bearer" or not supplied:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "authentication_required",
                    "message": "Bearer authentication is required",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        matched: AuthPrincipal | None = None
        for expected, principal in self.tokens:
            if hmac.compare_digest(supplied, expected):
                matched = principal
        if matched is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "invalid_token", "message": "Bearer token is invalid"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return matched


def require_role(role: str):
    async def dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> AuthPrincipal:
        authorization = (
            f"{credentials.scheme} {credentials.credentials}"
            if credentials is not None
            else None
        )
        principal = request.app.state.authenticator.authenticate(authorization)
        if role not in principal.roles and "admin" not in principal.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "insufficient_role",
                    "message": f"role {role} is required",
                },
            )
        return principal

    return dependency

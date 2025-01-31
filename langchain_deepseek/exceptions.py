# -*- coding: utf-8 -*-

from __future__ import annotations

import httpx
import requests
from typing import Any, Optional, Literal

__all__ = [
    "AuthenticationError",
    "BadRequestError",
]


class DeepSeekError(Exception):
    pass


class APIError(DeepSeekError):
    message: str = None
    request: Optional[httpx.Request] = None
    body: Any = None
    code: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str]

    def __init__(self, message: str, request: httpx.Request, *, body: object | None) -> None:
        self.message = message
        self.request = request
        self.body = body


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(self, message: str, *, response: httpx.Response, body: object | None) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(APIError):
    def __init__(self, *, message: str = "Connection error.", request: httpx.Request) -> None:
        super().__init__(message, request, body=None)


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="Request timed out.", request=request)


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400
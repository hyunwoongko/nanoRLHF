from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CallRequest:
    call_id: str
    method_name: str
    payload: bytes


@dataclass(frozen=True)
class ResizeRequest:
    new_max_concurrency: int


@dataclass(frozen=True)
class ShutdownRequest:
    pass


@dataclass(frozen=True)
class CreatedResponse:
    actor_id: str
    max_concurrency: int


@dataclass(frozen=True)
class ResizedResponse:
    actor_id: str
    max_concurrency: int


@dataclass(frozen=True)
class ResultResponse:
    actor_id: str
    call_id: str
    ok: bool
    value_payload: Optional[bytes]
    error_payload: Optional[bytes]


@dataclass(frozen=True)
class ShutdownDoneResponse:
    actor_id: str

"""Cancellation support for the active Streamlit analysis."""

from threading import Event, Lock


class PipelineCancelled(Exception):
    """Raised when an in-progress analysis is replaced by a refreshed page."""


class CancellationToken:
    def __init__(self):
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self._event.is_set():
            raise PipelineCancelled("Analysis stopped because the page was refreshed.")


_lock = Lock()
_active_token: CancellationToken | None = None


def cancel_active_run() -> None:
    with _lock:
        if _active_token:
            _active_token.cancel()


def start_run() -> CancellationToken:
    global _active_token
    with _lock:
        if _active_token:
            _active_token.cancel()
        _active_token = CancellationToken()
        return _active_token


def finish_run(token: CancellationToken) -> None:
    global _active_token
    with _lock:
        if _active_token is token:
            _active_token = None

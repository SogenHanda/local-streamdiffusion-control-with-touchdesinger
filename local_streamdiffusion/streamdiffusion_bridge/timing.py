"""Frame pacing without the coarse Windows threading.Event timeout tick."""
from __future__ import annotations

import ctypes
import sys
import threading


class FrameWaiter:
    """A receiver-thread-owned timer with short, bounded waits for shutdown."""

    def __init__(self) -> None:
        self._kernel = None
        self._timer = None
        if sys.platform != "win32":
            return
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateWaitableTimerExW.argtypes = (
            ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        )
        kernel.CreateWaitableTimerExW.restype = wintypes.HANDLE
        kernel.SetWaitableTimer.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong), wintypes.LONG,
            ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
        )
        kernel.SetWaitableTimer.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        # High-resolution waitable timers are supported from Windows 10 1803.
        # Unlike timeBeginPeriod, this does not depend on monitor visibility.
        timer = kernel.CreateWaitableTimerExW(None, None, 0x2, 0x100002)
        if timer:
            self._kernel, self._timer = kernel, timer

    def wait(self, seconds: float, stop_event: threading.Event) -> None:
        if seconds <= 0 or stop_event.is_set():
            return
        seconds = min(seconds, 0.05)
        if self._timer is not None:
            due = ctypes.c_longlong(-max(1, round(seconds * 10_000_000)))
            if self._kernel.SetWaitableTimer(self._timer, ctypes.byref(due), 0, None, None, False):
                # Bound even a failed timer so shutdown cannot hang indefinitely.
                if self._kernel.WaitForSingleObject(self._timer, 100) == 0:
                    return
            self.close()
        stop_event.wait(seconds)

    def close(self) -> None:
        if self._timer is not None:
            self._kernel.CloseHandle(self._timer)
            self._timer = None

    def __enter__(self) -> "FrameWaiter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

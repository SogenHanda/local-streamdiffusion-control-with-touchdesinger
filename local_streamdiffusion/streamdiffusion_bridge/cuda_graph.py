"""CUDA graph capture for stateless TinyVAE encode/decode calls."""
from __future__ import annotations

from typing import Any, Callable


class CapturedVAECall:
    """Keep inputs/outputs alive and recapture when shape or options change.

    The returned output is borrowed until the next call. Only the stateless
    TinyVAE network belongs inside this graph; noise, scheduler state and
    temporal history remain outside and are updated on every frame.
    """

    def __init__(self, callback: Callable[..., Any]) -> None:
        self.callback = callback
        self._graph = None
        self._key = None
        self._stream = None
        self._input = None
        self._output = None

    def __call__(self, value: Any, **kwargs: Any) -> Any:
        import torch

        caller = torch.cuda.current_stream(value.device)
        key = (tuple(value.shape), value.dtype, value.device, tuple(sorted(kwargs.items())))
        if self._key is not None and key != self._key:
            self.close()
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=value.device)
        self._stream.wait_stream(caller)
        with torch.cuda.stream(self._stream):
            if self._graph is None:
                self._input = torch.empty_like(value)
                self._input.copy_(value)
                # Initialize cuDNN workspaces outside capture.
                for _ in range(3):
                    self.callback(self._input, **kwargs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self._stream):
                    self._output = self.callback(self._input, **kwargs)
                self._graph = graph
                self._key = key
            self._input.copy_(value)
            value.record_stream(self._stream)
            self._graph.replay()
        caller.wait_stream(self._stream)
        return self._output

    def close(self) -> None:
        if self._stream is not None:
            import torch
            torch.cuda.synchronize(self._stream.device)
        self._graph = None
        self._input = self._output = None
        self._key = self._stream = None

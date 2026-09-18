"""TensorRT UNet execution with reusable buffers and explicit stream ordering."""
from __future__ import annotations

from typing import Any


class BufferedTensorRTUNet:
    """Single-worker adapter. Output is borrowed until the next inference call."""

    def __init__(self, engine: Any, *, use_cuda_graph: bool = False) -> None:
        self.engine = engine
        self._buffer_key = None
        self.use_cuda_graph = use_cuda_graph
        self._cuda_graph = None
        self._stream = None

    def __call__(self, latent_model_input: Any, timestep: Any,
                 encoder_hidden_states: Any, **_kwargs: Any) -> Any:
        import torch
        from diffusers.models.unet_2d_condition import UNet2DConditionOutput

        timestep = timestep.float()
        inputs = {"sample": latent_model_input, "timestep": timestep,
                  "encoder_hidden_states": encoder_hidden_states}
        shape_dict = {name: tuple(value.shape) for name, value in inputs.items()}
        shape_dict["latent"] = tuple(latent_model_input.shape)
        key = tuple((name, shape, str(inputs[name].dtype) if name in inputs else "")
                    for name, shape in shape_dict.items()) + (str(latent_model_input.device),)
        if key != self._buffer_key:
            self.close()
            self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)
            for name, tensor in self.engine.tensors.items():
                if not self.engine.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT buffer binding failed: {name}")
            self._buffer_key = key

        caller = torch.cuda.current_stream(latent_model_input.device)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=latent_model_input.device)
        self._stream.wait_stream(caller)
        with torch.cuda.stream(self._stream):
            # Copy all three inputs on every call, outside graph capture. Prompt,
            # strength, seed and camera changes must never replay old values.
            for name, value in inputs.items():
                self.engine.tensors[name].copy_(value)
                value.record_stream(self._stream)
            if self.use_cuda_graph:
                if self._cuda_graph is None:
                    self._execute()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self._stream):
                        self._execute()
                    self._cuda_graph = graph
                self._cuda_graph.replay()
            else:
                self._execute()
        caller.wait_stream(self._stream)
        return UNet2DConditionOutput(sample=self.engine.tensors["latent"])

    def _execute(self) -> None:
        if not self.engine.context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT UNet execution failed")

    def to(self, *_args: Any, **_kwargs: Any) -> "BufferedTensorRTUNet":
        return self

    def close(self) -> None:
        if self._stream is not None:
            import torch
            # Reconfiguration/shutdown only: finish consumers of borrowed
            # outputs before freeing a graph pool or changing buffer addresses.
            torch.cuda.synchronize(self._stream.device)
        self._cuda_graph = None
        self._stream = None
        self._buffer_key = None

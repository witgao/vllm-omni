# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import chain
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class LayerwiseOffloadHook(ModelHook):
    """Hook for layerwise (transformer-block-wise) CPU offloading.

    The hook instance retains parameters for both the current registered block
    module and those for the next block, as well as flattened CPU tensors which
    record the parameters of the current block module, so that these parameters
    could be re-materialized on device in an overlapping way.
    This hook should be registered to each of the transformer blocks in DiT
    module(s) of the target pipeline.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    _HOOK_NAME = "layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        stream: current_omni_platform.Stream | None = None,
        pin_memory: bool = True,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.copy_stream = stream or current_omni_platform.current_stream()
        self.pin_memory = pin_memory

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: current_omni_platform.Event | None = None

        # Backward link to the hook that is responsible for prefetching *this* block's weights
        self._prev_hook: LayerwiseOffloadHook | None = None

        self.module: nn.Module | None = None
        self.dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        self.dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        # This all happen during the hook instance being registered to hook registry;
        # the input module is kept intact
        module = super().initialize_hook(module)
        self.module = module

        next_block_parameters = dict(self.next_block.named_parameters())
        next_block_buffers = dict(self.next_block.named_buffers())

        # Pre-allocate gpu tensors in a flattened way
        self.dtype_cpu_flattened_weights, self.dtype_metadata = LayerwiseOffloadHook._to_cpu(
            self.next_block,
            next_block_parameters,
            next_block_buffers,
            self.device,
            self.pin_memory,
        )
        return module

    @staticmethod
    def _get_named_params_and_bufs(module: nn.Module) -> tuple[dict[str, nn.Parameter], dict[str, torch.Tensor]]:
        return dict(module.named_parameters()), dict(module.named_buffers())

    @staticmethod
    def _is_tensor_subclass_tensor(tensor: torch.Tensor) -> bool:
        return hasattr(tensor, "__tensor_flatten__") and hasattr(type(tensor), "__tensor_unflatten__")

    @staticmethod
    def _is_materialized_tensor(tensor: torch.Tensor) -> bool:
        if LayerwiseOffloadHook._is_tensor_subclass_tensor(tensor):
            tensor_names, _ = tensor.__tensor_flatten__()
            return any(
                LayerwiseOffloadHook._is_materialized_tensor(getattr(tensor, inner_name)) for inner_name in tensor_names
            )
        return tensor.dim() > 0

    @staticmethod
    def _make_empty_tensor_subclass_like(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        tensor_names, attrs = tensor.__tensor_flatten__()
        tensor_data_dict: dict[str, torch.Tensor] = {}
        for inner_name in tensor_names:
            inner_tensor = getattr(tensor, inner_name)
            tensor_data_dict[inner_name] = torch.empty((), device=device, dtype=inner_tensor.dtype)
        return type(tensor).__tensor_unflatten__(tensor_data_dict, attrs, None, None)

    @staticmethod
    def _replace_named_param_or_buffer(
        module: nn.Module,
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        owner_name, local_name = name.rsplit(".", 1) if "." in name else ("", name)
        owner = module.get_submodule(owner_name) if owner_name else module

        if name in params:
            requires_grad = params[name].requires_grad
            tensor = nn.Parameter(tensor, requires_grad=requires_grad)
            owner._parameters[local_name] = tensor
            params[name] = tensor
            return

        owner._buffers[local_name] = tensor
        bufs[name] = tensor

    @staticmethod
    def _make_tensor_subclass_inner_key(name: str, inner_name: str) -> str:
        return f"{name}::__tensor_inner__::{inner_name}"

    @staticmethod
    def _split_tensor_subclass_inner_key(name: str) -> tuple[str, str] | None:
        separator = "::__tensor_inner__::"
        if separator not in name:
            return None
        return name.split(separator, 1)  # type: ignore[return-value]

    @staticmethod
    def _get_tensor_storage_span(tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.numel() == 0:
            return tensor.storage_offset(), 0

        strides = tensor.stride()
        if any(stride < 0 for stride in strides):
            raise ValueError(
                f"Negative stride tensor is not supported by layerwise offload: "
                f"shape={tuple(tensor.shape)}, stride={tuple(strides)}"
            )

        storage_offset = tensor.storage_offset()
        max_offset = storage_offset
        for size, stride in zip(tensor.shape, strides):
            max_offset += (size - 1) * stride

        return storage_offset, max_offset - storage_offset + 1

    @staticmethod
    def _to_cpu(
        module: nn.Module,
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        device: torch.device,
        pin_memory: bool = True,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Helper method to move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # NOTE: order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            if LayerwiseOffloadHook._is_tensor_subclass_tensor(param_or_buf):
                tensor_names, _ = param_or_buf.__tensor_flatten__()
                for inner_name in tensor_names:
                    inner_tensor = getattr(param_or_buf, inner_name)
                    dtype = inner_tensor.dtype
                    if dtype not in dtype_grouped_weights:
                        dtype_grouped_weights[dtype] = {}
                    combined_name = LayerwiseOffloadHook._make_tensor_subclass_inner_key(name, inner_name)
                    dtype_grouped_weights[dtype][combined_name] = inner_tensor

                LayerwiseOffloadHook._replace_named_param_or_buffer(
                    module,
                    params,
                    bufs,
                    name,
                    LayerwiseOffloadHook._make_empty_tensor_subclass_like(param_or_buf, device),
                )
                continue
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            total_numel = sum(LayerwiseOffloadHook._get_tensor_storage_span(t)[1] for _, t in name2weights.items())
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)

            current_offset = 0
            for name, param_or_buf in name2weights.items():
                storage_offset, storage_numel = LayerwiseOffloadHook._get_tensor_storage_span(param_or_buf)
                storage_view = param_or_buf.as_strided((storage_numel,), (1,), storage_offset)
                cpu_tensor[current_offset : current_offset + storage_numel].copy_(storage_view)
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "storage_numel": storage_numel,
                        "shape": tuple(param_or_buf.shape),
                        "stride": tuple(param_or_buf.stride()),
                    }
                )

                param_or_buf.data = torch.empty((), device=device, dtype=dtype)
                current_offset += storage_numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor
        return dtype_cpu_flattened_weights, dtype_metadata

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        if self.module is None:
            return True

        block_parameters, _ = self._get_named_params_and_bufs(self.module)
        if not block_parameters:
            return True
        return any(self._is_materialized_tensor(param.data) for param in block_parameters.values())

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target block in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())
        layer_params, layer_bufs = self._get_named_params_and_bufs(self.next_block)

        evt = current_omni_platform.Event()
        gpu_weights: dict[torch.dtype, torch.Tensor] = {}

        with current_omni_platform.stream(self.copy_stream):
            for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
                gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                gpu_weights[dtype] = gpu_weight

            evt.record(self.copy_stream)

        tensor_subclass_rebuilds: dict[str, dict[str, torch.Tensor]] = {}
        for dtype, ordered_metadata in self.dtype_metadata.items():
            # ordered_metadata: list[dict[str, Any]]
            gpu_weight = gpu_weights[dtype]

            for metadata in ordered_metadata:
                target_name = metadata["name"]
                storage_view = gpu_weight[metadata["offset"] : metadata["offset"] + metadata["storage_numel"]]
                weight_view = storage_view.as_strided(metadata["shape"], metadata["stride"])

                if target_name in layer_params or target_name in layer_bufs:
                    target_param_or_buf = (
                        layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                    )
                    target_param_or_buf.data = weight_view
                    continue

                split_name = LayerwiseOffloadHook._split_tensor_subclass_inner_key(target_name)
                if split_name is None:
                    raise KeyError(f"Unexpected weight name during prefetch: {target_name}")

                outer_name, inner_name = split_name
                tensor_subclass_rebuild = tensor_subclass_rebuilds.setdefault(outer_name, {})
                tensor_subclass_rebuild[inner_name] = weight_view

        for target_name, tensor_data_dict in tensor_subclass_rebuilds.items():
            target_param_or_buf = layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
            _, attrs = target_param_or_buf.__tensor_flatten__()
            rebuilt_tensor = type(target_param_or_buf).__tensor_unflatten__(
                tensor_data_dict,
                attrs,
                None,
                None,
            )
            LayerwiseOffloadHook._replace_named_param_or_buffer(
                self.next_block,
                layer_params,
                layer_bufs,
                target_name,
                rebuilt_tensor,
            )

        self._prefetch_done = evt

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders.
        This function does not actually offload weights from GPU back to CPU.
        """
        evt = self._prefetch_done
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)

        self._prefetch_done = None

        # free GPU residency
        if self.module is None:
            return
        block_parameters, block_buffers = self._get_named_params_and_bufs(self.module)

        for name, param in list(block_parameters.items()):
            if LayerwiseOffloadHook._is_tensor_subclass_tensor(param):
                LayerwiseOffloadHook._replace_named_param_or_buffer(
                    self.module,
                    block_parameters,
                    block_buffers,
                    name,
                    LayerwiseOffloadHook._make_empty_tensor_subclass_like(param, self.device),
                )
            else:
                param.data = torch.empty((), device=self.device, dtype=param.dtype)

        for name, buf in list(block_buffers.items()):
            if LayerwiseOffloadHook._is_tensor_subclass_tensor(buf):
                LayerwiseOffloadHook._replace_named_param_or_buffer(
                    self.module,
                    block_parameters,
                    block_buffers,
                    name,
                    LayerwiseOffloadHook._make_empty_tensor_subclass_like(buf, self.device),
                )
            else:
                buf.data = torch.empty((), device=self.device, dtype=buf.dtype)

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # if the previous hook was skipped and the weights are not on device,
        # (e.g. by cache-dit block caching), ask the previous hook to
        # synchronously prefetch *this* block's weights before computation
        if not self.is_materialized and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(non_blocking=False)

        self.prefetch_layer(non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()

        return output


def apply_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    stream: current_omni_platform.Stream | None = None,
    pin_memory: bool = True,
) -> LayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = LayerwiseOffloadHook(next_block, device, stream, pin_memory)
    registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)

    return hook


def remove_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(LayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self._blocks: list[list[nn.Module]] = []

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping layer-wise offloading")
            return

        # Move encoders to GPU (they stay resident)
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE to GPU if available
        if modules.vae is not None:
            try:
                modules.vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Apply block-wise offloading hook for each of the blocks in DiT model(s)
        # Note that there might exist multiple DiT models in specific pipelines
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_name = LayerWiseOffloadBackend.get_blocks_attr_name(dit_module)
            blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)

            if not blocks_attr_name or not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            num_blocks = len(blocks)
            if num_blocks <= 1:
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name == blocks_attr_name:
                    logger.debug(f"Skipped blocks module {name}")
                    continue
                m.to(self.device)
                logger.debug(f"Moved {name} to device {self.device}")

            # Pre-fetch the first layer by manually calling the hook function on the last layer;
            # For subsequent requests, the first layer/block will be pre-fetched
            # during the last layer compute of the previous request.
            last_block, first_block = blocks[-1], blocks[0]
            last_hook = apply_block_hook(
                last_block, first_block, self.device, self.copy_stream, self.config.pin_cpu_memory
            )
            last_hook.prefetch_layer(non_blocking=False)

            block_hooks: list[LayerwiseOffloadHook] = [last_hook]
            # Register hook for each of blocks
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                hook = apply_block_hook(block, next_block, self.device, self.copy_stream, self.config.pin_cpu_memory)
                block_hooks.append(hook)

            # NOTE(yuanheng-zhao): We make each hook gets a backward reference to the hook
            # that is responsible for prefetching its block's weights. This is specifically a
            # workaround for that arbitrary blocks are skipped by caching systems (e.g., cache-dit)
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            logger.info(f"Layer-wise offloading enabled on {num_blocks} layers (blocks)")

            # Track hooked blocks for cleanup
            self._blocks.append(blocks)

        if len(self._blocks) > 0 and len(self._blocks[0]) > 0:
            self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_block_hook(block)

        self._blocks.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")

    @staticmethod
    def get_blocks_attr_name(model: nn.Module) -> str | None:
        """Retrieve blocks attribute name from provided DiT model"""
        return getattr(model.__class__, "_layerwise_offload_blocks_attr", None)

    @staticmethod
    def set_blocks_attr_name(model: nn.Module, name: str) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attr"):
            setattr(model.__class__, "_layerwise_offload_blocks_attr", name)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> list[nn.Module]:
        """
        Retrieve a list of blocks from provided DiT model. Blocks attribute name
        are found by `_layerwise_offload_blocks_attr` set to DiT models. For example,

        ```
        class WanTransformer3DModel(nn.Module):
            _layerwise_offload_blocks_attr = "blocks"
        ```
        """
        blocks_attr_name = LayerWiseOffloadBackend.get_blocks_attr_name(model)
        if blocks_attr_name is None:
            logger.warning(
                f"No _layerwise_offload_blocks_attr defined for {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        _blocks = getattr(model, blocks_attr_name, None)
        if _blocks is None:
            logger.warning(
                f"Blocks (layers) '{blocks_attr_name}' not found on {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return []

        return list(_blocks)

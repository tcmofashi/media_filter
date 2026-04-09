"""Distributed training utilities for Stage 1.

This module provides utilities for converting single-GPU Stage 1 training
to distributed multi-GPU training using PyTorch DDP.
"""

import io
import pickle
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from torch.utils.data.distributed import DistributedSampler

from src.logger import get_logger

logger = get_logger(__name__)


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get current rank, returns 0 if not distributed."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get world size, returns 1 if not distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def setup_distributed_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> Optional[Sampler]:
    if not dist.is_initialized():
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap model with DDP if in distributed mode.

    Args:
        model: Model to wrap
        device_id: GPU device ID (uses current device if None)
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model if distributed, original model otherwise
    """
    if not dist.is_initialized():
        return model

    device_ids = [device_id] if device_id is not None else None
    output_device = device_id if device_id is not None else None

    wrapped = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
    )

    if is_main_process():
        logger.info(f"Model wrapped with DDP on rank {get_rank()}")

    return wrapped


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP to get underlying model.

    Args:
        model: Potentially DDP-wrapped model

    Returns:
        Unwrapped model
    """
    if isinstance(model, DDP):
        return model.module
    return model


def aggregate_scalar(value: float, op: str = "mean") -> float:
    """Aggregate scalar value across all ranks.

    Args:
        value: Scalar value to aggregate
        op: Aggregation operation ('mean' or 'sum')

    Returns:
        Aggregated value
    """
    if not dist.is_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float32, device="cuda")

    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unknown aggregation op: {op}")

    return tensor.item()


def aggregate_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """Aggregate tensor across all ranks in-place.

    Args:
        tensor: Tensor to aggregate (must be on CUDA)
        op: Aggregation operation ('mean' or 'sum')

    Returns:
        Aggregated tensor
    """
    if not dist.is_initialized():
        return tensor

    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unknown aggregation op: {op}")

    return tensor


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from src rank to all ranks.

    Args:
        tensor: Tensor to broadcast (only valid on src rank)
        src: Source rank

    Returns:
        Broadcast tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast arbitrary Python object from src rank to all ranks.

    Uses pickle serialization. For large objects (like hidden_states_dict),
    consider using broadcast_dict_with_disk for memory efficiency.

    Args:
        obj: Object to broadcast (only valid on src rank)
        src: Source rank

    Returns:
        Broadcast object
    """
    if not dist.is_initialized():
        return obj

    rank = get_rank()

    if rank == src:
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        data = buffer.getvalue()
        size = torch.tensor(len(data), dtype=torch.long, device="cuda")
    else:
        data = b""
        size = torch.tensor(0, dtype=torch.long, device="cuda")

    dist.broadcast(size, src=src)

    size_int = int(size.item())

    if rank != src:
        data = bytes(size_int)

    tensor = torch.ByteTensor(list(data) if rank == src else [0] * size_int).cuda()
    dist.broadcast(tensor, src=src)

    if rank != src:
        buffer = io.BytesIO(bytes(tensor.cpu().numpy()))
        obj = pickle.load(buffer)

    return obj


def broadcast_dict(obj_dict: Dict[str, Any], src: int = 0) -> Dict[str, Any]:
    """Broadcast dictionary from src rank to all ranks.

    Warning: For large dictionaries with tensors, this may be slow.
    Consider using disk-based approach for hidden_states_dict.

    Args:
        obj_dict: Dictionary to broadcast (only valid on src rank)
        src: Source rank

    Returns:
        Broadcast dictionary
    """
    return broadcast_object(obj_dict, src=src)


def broadcast_state_dict(
    state_dict: Dict[str, torch.Tensor], src: int = 0
) -> Dict[str, torch.Tensor]:
    """Broadcast model state dict from src rank to all ranks.

    More efficient than broadcast_dict for state dicts with tensors.

    Args:
        state_dict: State dict to broadcast (only valid on src rank)
        src: Source rank

    Returns:
        Broadcast state dict
    """
    if not dist.is_initialized():
        return state_dict

    rank = get_rank()

    # Get keys on all ranks
    if rank == src:
        keys = list(state_dict.keys())
    else:
        keys = None
    keys = broadcast_object(keys, src=src)

    # Broadcast each tensor
    result: Dict[str, torch.Tensor] = {}
    for key in keys:
        if rank == src:
            src_tensor = state_dict[key].clone().cuda()
            shape = torch.tensor(src_tensor.shape, dtype=torch.long, device="cuda")
            dtype_code = torch.tensor(
                _dtype_to_code(src_tensor.dtype), dtype=torch.long, device="cuda"
            )
        else:
            shape = torch.tensor([], dtype=torch.long, device="cuda")
            dtype_code = torch.tensor(0, dtype=torch.long, device="cuda")
            src_tensor = None

        dist.broadcast(shape, src=src)
        dist.broadcast(dtype_code, src=src)

        if rank == src and src_tensor is not None:
            tensor = src_tensor
        else:
            dtype = _code_to_dtype(int(dtype_code.item()))
            tensor = torch.empty(tuple(shape.tolist()), dtype=dtype, device="cuda")

        dist.broadcast(tensor, src=src)
        result[key] = tensor

    return result


def _dtype_to_code(dtype: torch.dtype) -> int:
    """Convert torch dtype to integer code."""
    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
        torch.uint8: 5,
    }
    return dtype_map.get(dtype, 0)


def _code_to_dtype(code: int) -> torch.dtype:
    """Convert integer code to torch dtype."""
    code_map = {
        0: torch.float32,
        1: torch.float16,
        2: torch.bfloat16,
        3: torch.int64,
        4: torch.int32,
        5: torch.uint8,
    }
    return code_map.get(code, torch.float32)


def verify_gradient_sync(model: torch.nn.Module, rtol: float = 1e-5) -> bool:
    """Verify that gradients are synchronized across all ranks.

    This is a diagnostic function to ensure DDP is working correctly.
    Should be called after backward() but before optimizer.step().

    Args:
        model: DDP-wrapped model
        rtol: Relative tolerance for comparison

    Returns:
        True if gradients are synchronized, False otherwise
    """
    if not dist.is_initialized():
        return True

    model = unwrap_ddp(model)

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Get local gradient norm
            local_norm = param.grad.norm().item()

            # Aggregate across ranks
            global_norm = torch.tensor(local_norm, device="cuda")
            dist.all_reduce(global_norm, op=dist.ReduceOp.SUM)
            global_norm = global_norm / dist.get_world_size()

            # Check if local matches global (within tolerance)
            if abs(local_norm - global_norm.item()) > rtol * max(
                abs(local_norm), abs(global_norm.item())
            ):
                logger.warning(
                    f"Gradient sync mismatch for {name}: local={local_norm:.6f}, global={global_norm.item():.6f}"
                )
                return False

    return True


def verify_weights_sync(model: torch.nn.Module, rtol: float = 1e-5) -> bool:
    """Verify that weights are synchronized across all ranks.

    This is a diagnostic function to ensure all ranks have identical model weights.

    Args:
        model: Model to check
        rtol: Relative tolerance for comparison

    Returns:
        True if weights are synchronized, False otherwise
    """
    if not dist.is_initialized():
        return True

    model = unwrap_ddp(model)

    for name, param in model.named_parameters():
        # Get local weight norm
        local_norm = param.data.norm().item()

        # Aggregate across ranks
        global_norm = torch.tensor(local_norm, device="cuda")
        dist.all_reduce(global_norm, op=dist.ReduceOp.SUM)
        global_norm = global_norm / dist.get_world_size()

        # Check if local matches global (within tolerance)
        if abs(local_norm - global_norm.item()) > rtol * max(
            abs(local_norm), abs(global_norm.item())
        ):
            logger.warning(
                f"Weight sync mismatch for {name}: local={local_norm:.6f}, global={global_norm.item():.6f}"
            )
            return False

    return True

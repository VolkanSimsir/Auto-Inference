"""
Utility functions for Auto-Inference.

Provides helpers for device detection, memory reporting, and model metadata.
All functions are pure utilities with no side effects on library state.
"""

from typing import Dict, Optional


def get_device_info() -> Dict[str, object]:
    """
    Return a summary of available compute devices.

    Returns:
        Dictionary with keys:
            - ``cuda_available`` (bool): Whether CUDA is available.
            - ``cuda_device_count`` (int): Number of CUDA GPUs.
            - ``cuda_devices`` (list[dict]): Per-device name and total memory (GiB).
            - ``mps_available`` (bool): Whether Apple MPS is available.
            - ``recommended_device`` (str): Suggested device string for model loading.

    Example:
        >>> from auto_inference.utils import get_device_info
        >>> info = get_device_info()
        >>> print(info["recommended_device"])
        'cuda'
    """
    import torch

    info: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
        "mps_available": torch.backends.mps.is_available(),
        "recommended_device": "cpu",
    }

    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / (1024**3), 2),
                }
            )
        info["cuda_devices"] = devices
        info["recommended_device"] = "auto"
    elif info["mps_available"]:
        info["recommended_device"] = "mps"

    return info


def get_memory_usage() -> Dict[str, float]:
    """
    Return current GPU and CPU memory usage in GiB.

    Returns:
        Dictionary with:
            - ``gpu_allocated_gib``: Allocated GPU memory in GiB (0 if no GPU).
            - ``gpu_reserved_gib``: Reserved (cached) GPU memory in GiB.
            - ``gpu_free_gib``: Remaining free GPU memory in GiB.
            - ``ram_used_gib``: Current process RAM usage in GiB.
            - ``ram_total_gib``: Total system RAM in GiB.

    Example:
        >>> from auto_inference.utils import get_memory_usage
        >>> mem = get_memory_usage()
        >>> print(f"GPU: {mem['gpu_allocated_gib']:.1f} GiB allocated")
    """
    import torch

    result: Dict[str, float] = {
        "gpu_allocated_gib": 0.0,
        "gpu_reserved_gib": 0.0,
        "gpu_free_gib": 0.0,
        "ram_used_gib": 0.0,
        "ram_total_gib": 0.0,
    }

    if torch.cuda.is_available():
        result["gpu_allocated_gib"] = torch.cuda.memory_allocated() / (1024**3)
        result["gpu_reserved_gib"] = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result["gpu_free_gib"] = round(total - result["gpu_reserved_gib"], 2)

    try:
        import psutil

        proc = psutil.Process()
        result["ram_used_gib"] = proc.memory_info().rss / (1024**3)
        result["ram_total_gib"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass  # psutil is optional

    return {k: round(v, 3) for k, v in result.items()}


def list_hf_models(
    search: str,
    task: str = "text-generation",
    limit: int = 10,
) -> list:
    """
    Search HuggingFace Hub for models matching a query.

    Requires the ``huggingface_hub`` package (included in core requirements).

    Args:
        search: Free-text search query (e.g. 'llama', 'mistral instruct').
        task: HuggingFace task filter. Defaults to 'text-generation'.
        limit: Maximum number of results to return. Defaults to 10.

    Returns:
        List of dicts with keys ``model_id``, ``downloads``, ``likes``,
        and ``last_modified``.

    Example:
        >>> from auto_inference.utils import list_hf_models
        >>> for m in list_hf_models("mistral 7b", limit=5):
        ...     print(m["model_id"], m["downloads"])
    """
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(
        search=search,
        filter=task,
        sort="downloads",
        direction=-1,
        limit=limit,
    )

    results = []
    for m in models:
        results.append(
            {
                "model_id": m.modelId,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "last_modified": str(getattr(m, "lastModified", "")),
            }
        )
    return results


def estimate_model_memory(
    model_id: str,
    dtype: str = "float16",
) -> Optional[float]:
    """
    Estimate how much GPU memory (in GiB) a model needs.

    Uses the model's parameter count from the HuggingFace Hub API.
    This is an approximation; actual usage may vary due to KV cache,
    activations, and other overheads.

    Args:
        model_id: HuggingFace model ID (e.g. 'meta-llama/Meta-Llama-3-8B').
        dtype: Data type for estimation: 'float32' (4 bytes),
            'float16' or 'bfloat16' (2 bytes), 'int8' (1 byte),
            'int4' or 'nf4' (0.5 bytes). Defaults to 'float16'.

    Returns:
        Estimated memory in GiB, or None if metadata is unavailable.

    Example:
        >>> from auto_inference.utils import estimate_model_memory
        >>> gib = estimate_model_memory("meta-llama/Meta-Llama-3-8B", "float16")
        >>> print(f"~{gib:.1f} GiB needed")
    """
    bytes_per_param = {
        "float32": 4.0,
        "float16": 2.0,
        "bfloat16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "nf4": 0.5,
    }

    bpp = bytes_per_param.get(dtype.lower())
    if bpp is None:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Choose from: {list(bytes_per_param)}"
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(model_id)
        params = getattr(info, "safetensors", None)
        if params and hasattr(params, "total"):
            return round(params.total * bpp / (1024**3), 2)
    except Exception:
        pass

    return None


def print_system_info() -> None:
    """
    Print a formatted summary of the system's compute environment.

    Displays Python version, PyTorch version, CUDA availability, GPU specs,
    and current memory usage.

    Example:
        >>> from auto_inference.utils import print_system_info
        >>> print_system_info()
    """
    import sys

    import torch

    print("=" * 50)
    print("Auto-Inference System Info")
    print("=" * 50)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    device_info = get_device_info()
    if device_info["cuda_available"]:
        print(f"CUDA:    available ({device_info['cuda_device_count']} GPU(s))")
        for dev in device_info["cuda_devices"]:  # type: ignore[union-attr]
            print(f"  GPU {dev['index']}: {dev['name']} — {dev['total_memory_gib']} GiB")
    elif device_info["mps_available"]:
        print("MPS:     available (Apple Silicon)")
    else:
        print("GPU:     not available (CPU only)")

    print(f"Recommended device: {device_info['recommended_device']}")

    mem = get_memory_usage()
    if mem["gpu_allocated_gib"] > 0:
        print(
            f"GPU memory: {mem['gpu_allocated_gib']:.2f} GiB allocated, "
            f"{mem['gpu_free_gib']:.2f} GiB free"
        )
    if mem["ram_total_gib"] > 0:
        print(
            f"RAM: {mem['ram_used_gib']:.2f} GiB used / {mem['ram_total_gib']:.1f} GiB total"
        )
    print("=" * 50)

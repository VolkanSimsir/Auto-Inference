from auto_inference.backends.base import BaseBackend, GenerationConfig

__all__ = [
    "BaseBackend",
    "GenerationConfig",
    "TransformersBackend",
    "UnslothBackend",
    "VLLMBackend",
    "LlamaCppBackend",
]


def __getattr__(name: str):
    """Lazy-load backend classes only when accessed."""
    _map = {
        "TransformersBackend": ("auto_inference.backends.transformers_backend", "TransformersBackend"),
        "UnslothBackend": ("auto_inference.backends.unsloth_backend", "UnslothBackend"),
        "VLLMBackend": ("auto_inference.backends.vllm_backend", "VLLMBackend"),
        "LlamaCppBackend": ("auto_inference.backends.llamacpp_backend", "LlamaCppBackend"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise AttributeError(f"module 'auto_inference.backends' has no attribute '{name}'")

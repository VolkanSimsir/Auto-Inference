from auto_inference.backends.base import BaseBackend, GenerationConfig
from auto_inference.core import AutoInference

__version__ = "0.3.0"
__author__ = "Volkan Simsir"
__email__ = ""
__license__ = "Apache-2.0"

__all__ = [
    # Main factory
    "AutoInference",
    # Base classes
    "BaseBackend",
    "GenerationConfig",
    # Backend classes (lazy-loaded on access)
    "TransformersBackend",
    "UnslothBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
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
    raise AttributeError(f"module 'auto_inference' has no attribute '{name}'")

from auto_inference.backends import (
    BaseBackend,
    GenerationConfig,
    LlamaCppBackend,
    TransformersBackend,
    UnslothBackend,
    VLLMBackend,
)
from auto_inference.core import AutoInference

__version__ = "0.2.0"
__author__ = "Kadir Nar"
__email__ = "kadir.nar@example.com"
__license__ = "Apache-2.0"

__all__ = [
    # Main factory
    "AutoInference",
    # Backend classes
    "BaseBackend",
    "TransformersBackend",
    "UnslothBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    # Configuration
    "GenerationConfig",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationConfig:
    """
    Configuration class for text generation parameters.

    This dataclass holds all common parameters used across different inference backends.
    Backend-specific parameters can be passed via the `extra_params` field.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate. Defaults to 512.
        temperature: Sampling temperature. Higher values produce more random outputs.
            Must be between 0 and 2. Defaults to 0.7.
        do_sample: Whether to use sampling instead of greedy decoding. Defaults to True.
        top_k: Top-k sampling parameter. Only the top k tokens are considered.
            Defaults to 50.
        top_p: Top-p (nucleus) sampling parameter. Tokens with cumulative probability
            up to top_p are considered. Defaults to 0.95.
        skip_special_tokens: Whether to skip special tokens in the output.
            Defaults to True.
        extra_params: Dictionary for backend-specific parameters.

    Example:
        >>> config = GenerationConfig(
        ...     max_new_tokens=256,
        ...     temperature=0.8,
        ...     top_p=0.9
        ... )
        >>> print(config.max_new_tokens)
        256
    """

    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    skip_special_tokens: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 (exclusive) and 1 (inclusive)")


class BaseBackend(ABC):
    """
    Abstract base class for all inference backends.

    This class defines the interface that all inference backends must implement.
    It provides a consistent API for loading models and generating text across
    different inference engines (Transformers, vLLM, Unsloth, Llama.cpp, etc.).

    All backends must implement:
        - `_load_model()`: Load and initialize the model
        - `generate()`: Generate text from a prompt
        - `__call__()`: Allow calling the instance directly for generation

    Attributes:
        model_name: The name or path of the model to load.
        config: Generation configuration parameters.
        model: The loaded model instance (set by `_load_model()`).
        tokenizer: The tokenizer instance if applicable (set by `_load_model()`).

    Example:
        >>> class MyBackend(BaseBackend):
        ...     def _load_model(self) -> None:
        ...         self.model = load_my_model(self.model_name)
        ...
        ...     def generate(self, prompt: str) -> str:
        ...         return self.model.generate(prompt)
        ...
        >>> backend = MyBackend("my-model", GenerationConfig())
        >>> response = backend("Hello, world!")
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the backend.

        Args:
            model_name: The name or path of the model to load.
                Can be a HuggingFace model ID, local path, or backend-specific identifier.
            config: Generation configuration. If None, uses default GenerationConfig.
            **kwargs: Additional backend-specific parameters.

        Raises:
            ValueError: If model_name is empty or None.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.model: Any = None
        self.tokenizer: Any = None
        self._kwargs = kwargs

        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load and initialize the model.

        This method must be implemented by all subclasses. It should:
        1. Load the model from the specified path/name
        2. Initialize the tokenizer if needed
        3. Set up any backend-specific configurations

        The loaded model should be stored in `self.model` and the tokenizer
        (if applicable) in `self.tokenizer`.

        Raises:
            RuntimeError: If the model fails to load.
            ImportError: If required dependencies are not installed.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input text prompt for generation.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If generation fails.
        """
        pass

    def __call__(self, prompt: str) -> str:
        """
        Generate text by calling the instance directly.

        This method provides a convenient way to generate text by treating
        the backend instance as a callable.

        Args:
            prompt: The input text prompt for generation.

        Returns:
            The generated text response.

        Example:
            >>> backend = MyBackend("model-name")
            >>> response = backend("What is AI?")
            >>> print(response)
            "AI stands for Artificial Intelligence..."
        """
        return self.generate(prompt)

    def __repr__(self) -> str:
        """Return a string representation of the backend."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"max_new_tokens={self.config.max_new_tokens}, "
            f"temperature={self.config.temperature})"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

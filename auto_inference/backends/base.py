from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional


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
        repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage
            repetition. Defaults to 1.0 (no penalty).
        skip_special_tokens: Whether to skip special tokens in the output.
            Defaults to True.
        stop_sequences: List of strings that stop generation when encountered.
            Defaults to empty list (no stop sequences).
        extra_params: Dictionary for backend-specific parameters.

    Example:
        >>> config = GenerationConfig(
        ...     max_new_tokens=256,
        ...     temperature=0.8,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1,
        ...     stop_sequences=["</s>", "Human:"]
        ... )
    """

    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    skip_special_tokens: bool = True
    stop_sequences: List[str] = field(default_factory=list)
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
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


# Type alias for a chat message
Message = Dict[str, str]


class BaseBackend(ABC):
    """
    Abstract base class for all inference backends.

    This class defines the interface that all inference backends must implement.
    It provides a consistent API for loading models and generating text across
    different inference engines (Transformers, vLLM, Unsloth, Llama.cpp, etc.).

    All backends must implement:
        - `_load_model()`: Load and initialize the model
        - `generate()`: Generate text from a prompt

    Optional overrides:
        - `generate_stream()`: Streaming generation (yields tokens incrementally)
        - `generate_batch()`: Batch generation for multiple prompts
        - `chat()`: Multi-turn chat using model's chat template

    Attributes:
        model_name: The name or path of the model to load.
        config: Generation configuration parameters.
        model: The loaded model instance (set by `_load_model()`).
        tokenizer: The tokenizer instance if applicable (set by `_load_model()`).
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

        Must be implemented by subclasses. Should set `self.model` and
        optionally `self.tokenizer`.
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

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Generate text incrementally, yielding tokens as they are produced.

        Backends that support streaming should override this method.
        The default implementation falls back to `generate()` and yields the
        full result as a single chunk.

        Args:
            prompt: The input text prompt for generation.

        Yields:
            Text chunks as they are generated.

        Example:
            >>> for chunk in backend.generate_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        yield self.generate(prompt)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for a list of prompts.

        Backends that support efficient batching should override this method.
        The default implementation calls `generate()` sequentially.

        Args:
            prompts: List of input prompts.

        Returns:
            List of generated text responses, in the same order as the input.

        Example:
            >>> responses = backend.generate_batch(["Hello", "Hi", "Hey"])
        """
        return [self.generate(p) for p in prompts]

    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for a multi-turn conversation.

        Uses the model's chat template (via `tokenizer.apply_chat_template`)
        when available, otherwise formats messages into a plain text prompt.

        Backends with native chat support (e.g. TransformersBackend) override
        this for proper template handling.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Roles should be 'user', 'assistant', or 'system'.
            system_prompt: Optional system prompt prepended to the conversation.
                If provided, it is inserted as the first message with role='system'.

        Returns:
            The model's response text.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is Python?"},
            ... ]
            >>> response = backend.chat(messages, system_prompt="You are helpful.")
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        # Try to use the tokenizer's chat template if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self.generate(prompt)
            except Exception:
                pass  # Fall back to manual formatting

        # Manual fallback: concatenate messages into a readable prompt
        parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return self.generate("\n".join(parts))

    def __call__(self, prompt: str) -> str:
        """Allow calling the backend instance directly for generation."""
        return self.generate(prompt)

    def __repr__(self) -> str:
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

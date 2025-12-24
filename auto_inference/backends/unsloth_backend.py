from typing import Any, Optional

import torch

from auto_inference.backends.base import BaseBackend, GenerationConfig


class UnslothBackend(BaseBackend):
    """
    Inference backend using Unsloth for optimized LLM inference.

    Unsloth provides significant memory savings and speed improvements
    through optimized kernels and native 4-bit quantization support.
    It's particularly well-suited for fine-tuned models and inference
    on consumer-grade GPUs.

    Attributes:
        model_name: HuggingFace model ID or local path.
        config: Generation configuration parameters.
        max_seq_length: Maximum sequence length for the model.
        load_in_4bit: Whether to load the model in 4-bit precision.
        model: The loaded FastLanguageModel instance.
        tokenizer: The loaded tokenizer instance.

    Example:
        Basic usage with 4-bit quantization:

        >>> from auto_inference.backends import UnslothBackend
        >>> from auto_inference.backends.base import GenerationConfig
        >>>
        >>> backend = UnslothBackend(
        ...     model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        ...     config=GenerationConfig(max_new_tokens=256),
        ...     load_in_4bit=True
        ... )
        >>> response = backend("Explain the theory of relativity")
        >>> print(response)

    Example:
        Using with a fine-tuned model:

        >>> backend = UnslothBackend(
        ...     model_name="your-username/your-finetuned-model",
        ...     config=GenerationConfig(
        ...         max_new_tokens=512,
        ...         temperature=0.8,
        ...         do_sample=True
        ...     ),
        ...     max_seq_length=4096,
        ...     load_in_4bit=True
        ... )
        >>> response = backend("Write a poem about AI")

    Note:
        - Requires `unsloth` package to be installed.
        - CUDA-capable GPU is required (Unsloth is optimized for NVIDIA GPUs).
        - 4-bit quantization significantly reduces VRAM usage.
        - Works best with models trained or fine-tuned using Unsloth.

    Warning:
        Unsloth currently requires CUDA. CPU inference is not supported.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Unsloth backend.

        Args:
            model_name: HuggingFace model ID or local path to the model.
                Unsloth provides optimized versions of popular models with
                the naming convention 'unsloth/model-name-bnb-4bit'.
            config: Generation configuration. Uses defaults if not provided.
            max_seq_length: Maximum sequence length the model can handle.
                Should match the model's training configuration. Defaults to 2048.
            load_in_4bit: Whether to load the model with 4-bit quantization.
                Significantly reduces memory usage with minimal quality loss.
                Defaults to True.
            **kwargs: Additional arguments passed to FastLanguageModel.

        Raises:
            ImportError: If unsloth is not installed.
            RuntimeError: If CUDA is not available.
        """
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        super().__init__(model_name, config, **kwargs)

    def _load_model(self) -> None:
        """
        Load the model using Unsloth's FastLanguageModel.

        This method:
        1. Verifies CUDA availability
        2. Loads the model with specified quantization settings
        3. Applies inference optimizations
        4. Configures the tokenizer

        Raises:
            ImportError: If unsloth package is not installed.
            RuntimeError: If CUDA is not available.
            OSError: If the model cannot be loaded.
        """
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Unsloth backend requires CUDA. "
                "Please use TransformersBackend or LlamaCppBackend for CPU inference."
            )

        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "unsloth package is required for UnslothBackend. "
                "Install it with: pip install unsloth"
            ) from e

        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            **self._kwargs,
        )

        # Apply inference optimizations
        FastLanguageModel.for_inference(self.model)

        # Configure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        """
        Generate text from the given prompt.

        Args:
            prompt: The input text to generate a response for.

        Returns:
            Generated text response (without the original prompt).

        Raises:
            ValueError: If prompt is empty or None.
            RuntimeError: If generation fails.

        Example:
            >>> backend = UnslothBackend("unsloth/llama-3-8b-bnb-4bit")
            >>> response = backend.generate("What is deep learning?")
            >>> print(response)
            "Deep learning is a subset of machine learning..."
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Tokenize and move to GPU
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to("cuda")

        # Generate with no_grad for memory efficiency
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
                top_k=self.config.top_k if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=self.config.skip_special_tokens,
        )

        return response.strip()

    @property
    def memory_footprint(self) -> str:
        """Get approximate GPU memory usage information."""
        if not torch.cuda.is_available():
            return "CUDA not available"
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"

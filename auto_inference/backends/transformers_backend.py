from typing import Any, Literal, Optional

import torch

from auto_inference.backends.base import BaseBackend, GenerationConfig


class TransformersBackend(BaseBackend):
    """
    Inference backend using HuggingFace Transformers.

    This backend provides flexible inference using the Transformers library,
    supporting automatic device placement, mixed precision, and various
    generation strategies.

    Attributes:
        model_name: HuggingFace model ID or local path.
        config: Generation configuration parameters.
        device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps').
        dtype: Model data type for inference.
        model: The loaded AutoModelForCausalLM instance.
        tokenizer: The loaded AutoTokenizer instance.

    Example:
        Basic usage with a HuggingFace model:

        >>> from auto_inference.backends import TransformersBackend
        >>> from auto_inference.backends.base import GenerationConfig
        >>>
        >>> config = GenerationConfig(max_new_tokens=128, temperature=0.7)
        >>> backend = TransformersBackend(
        ...     model_name="meta-llama/Llama-2-7b-chat-hf",
        ...     config=config,
        ...     device="auto"
        ... )
        >>> response = backend("What is machine learning?")
        >>> print(response)

    Example:
        Using a local model with custom parameters:

        >>> backend = TransformersBackend(
        ...     model_name="/path/to/local/model",
        ...     config=GenerationConfig(
        ...         max_new_tokens=256,
        ...         temperature=0.9,
        ...         top_p=0.95,
        ...         do_sample=True
        ...     ),
        ...     device="cuda:0",
        ...     torch_dtype=torch.bfloat16
        ... )
        >>> response = backend("Explain quantum computing")

    Note:
        - Requires `transformers` and `torch` packages to be installed.
        - For large models, ensure sufficient GPU memory is available.
        - Use `device="auto"` for automatic device mapping across multiple GPUs.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Transformers backend.

        Args:
            model_name: HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')
                or path to a local model directory.
            config: Generation configuration. Uses defaults if not provided.
            device: Device for inference. Options:
                - 'auto': Automatic device mapping (recommended for multi-GPU)
                - 'cuda': Use default CUDA device
                - 'cuda:N': Use specific CUDA device
                - 'cpu': Use CPU only
                - 'mps': Use Apple Silicon GPU
            torch_dtype: Model precision. If None, uses float16 for GPU, float32 for CPU.
                Common options: torch.float16, torch.bfloat16, torch.float32
            trust_remote_code: Whether to trust and execute remote code from the model
                repository. Required for some models. Defaults to False for security.
            **kwargs: Additional arguments passed to model loading.

        Raises:
            ImportError: If transformers or torch is not installed.
            ValueError: If the model cannot be found or loaded.
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        super().__init__(model_name, config, **kwargs)

    def _load_model(self) -> None:
        """
        Load the model and tokenizer from HuggingFace.

        This method initializes:
        1. AutoTokenizer with proper padding configuration
        2. AutoModelForCausalLM with specified device and dtype

        Raises:
            ImportError: If required packages are not installed.
            OSError: If the model or tokenizer cannot be loaded.
            RuntimeError: If CUDA is requested but not available.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers package is required for TransformersBackend. "
                "Install it with: pip install transformers"
            ) from e

        # Determine dtype based on device
        if self.torch_dtype is None:
            if self.device == "cpu":
                self.torch_dtype = torch.float32
            else:
                self.torch_dtype = torch.float16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Configure padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare model loading arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            **self._kwargs,
        }

        # Configure device mapping
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Move to device if not using auto device map
        if self.device != "auto" and self.device != "cpu":
            self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

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
            >>> backend = TransformersBackend("gpt2")
            >>> response = backend.generate("Once upon a time")
            >>> print(response)
            " there was a princess who lived in a castle..."
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move inputs to the same device as model
        if self.device == "auto":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        # Decode and return only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=self.config.skip_special_tokens,
        )

        return response.strip()

    @property
    def device_info(self) -> str:
        """Get information about the current device configuration."""
        if self.model is None:
            return "Model not loaded"
        if hasattr(self.model, "hf_device_map"):
            return f"Multi-device: {self.model.hf_device_map}"
        return f"Device: {next(self.model.parameters()).device}"

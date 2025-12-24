from typing import Any, Dict, Literal, Optional, Type, Union

from auto_inference.backends.base import BaseBackend, GenerationConfig


# Type alias for supported backend names
BackendType = Literal["transformers", "unsloth", "vllm", "llamacpp"]


class AutoInference:
    """
    Factory class for creating inference backends.

    AutoInference provides a unified interface for loading and using different
    LLM inference backends. It abstracts away the complexity of each backend,
    allowing users to switch between engines with minimal code changes.

    Supported Backends:
        - 'transformers': HuggingFace Transformers (flexible, widely compatible)
        - 'unsloth': Unsloth (memory-efficient, 4-bit quantization)
        - 'vllm': vLLM (high-throughput, production-ready)
        - 'llamacpp': llama.cpp (CPU-optimized, edge deployment)

    Example:
        Basic usage with Transformers backend:

        >>> from auto_inference import AutoInference
        >>>
        >>> model = AutoInference.from_pretrained(
        ...     model_name="meta-llama/Llama-2-7b-chat-hf",
        ...     backend="transformers",
        ...     max_new_tokens=256,
        ...     temperature=0.7
        ... )
        >>> response = model("What is machine learning?")
        >>> print(response)

    Example:
        Using vLLM for high-throughput inference:

        >>> model = AutoInference.from_pretrained(
        ...     model_name="mistralai/Mistral-7B-v0.1",
        ...     backend="vllm",
        ...     max_new_tokens=512,
        ...     tensor_parallel_size=2
        ... )
        >>> response = model("Explain neural networks")

    Example:
        Memory-efficient inference with Unsloth:

        >>> model = AutoInference.from_pretrained(
        ...     model_name="unsloth/llama-3-8b-bnb-4bit",
        ...     backend="unsloth",
        ...     load_in_4bit=True,
        ...     max_new_tokens=256
        ... )
        >>> response = model("Write a Python function")

    Example:
        CPU inference with llama.cpp:

        >>> model = AutoInference.from_pretrained(
        ...     model_name="/path/to/model.gguf",
        ...     backend="llamacpp",
        ...     n_gpu_layers=0,  # Pure CPU
        ...     n_ctx=4096
        ... )
        >>> response = model("Hello!")

    Note:
        The factory returns an instance of the appropriate backend class,
        not an AutoInference instance. This allows direct access to
        backend-specific features while maintaining a consistent API.
    """

    # Registry mapping backend names to their implementation classes
    _BACKEND_REGISTRY: Dict[str, str] = {
        "transformers": "TransformersBackend",
        "unsloth": "UnslothBackend",
        "vllm": "VLLMBackend",
        "llamacpp": "LlamaCppBackend",
    }

    # Module paths for each backend
    _MODULE_PATHS: Dict[str, str] = {
        "transformers": "auto_inference.backends.transformers_backend",
        "unsloth": "auto_inference.backends.unsloth_backend",
        "vllm": "auto_inference.backends.vllm_backend",
        "llamacpp": "auto_inference.backends.llamacpp_backend",
    }

    def __init__(self) -> None:
        """
        Initialize AutoInference.

        Note:
            This class is not meant to be instantiated directly.
            Use the `from_pretrained` class method instead.
        """
        raise RuntimeError(
            "AutoInference should not be instantiated directly. "
            "Use AutoInference.from_pretrained() instead."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        backend: BackendType = "transformers",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> BaseBackend:
        """
        Create an inference backend from a pretrained model.

        This factory method loads the specified model using the chosen backend
        and returns an initialized backend instance ready for inference.

        Args:
            model_name: Model identifier or path. This can be:
                - HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')
                - Local path to model directory or file
                - For llamacpp: path to GGUF file
            backend: Backend to use for inference. One of:
                - 'transformers': HuggingFace Transformers
                - 'unsloth': Unsloth (requires CUDA)
                - 'vllm': vLLM (requires CUDA)
                - 'llamacpp': llama.cpp (CPU/GPU)
                Defaults to 'transformers'.
            max_new_tokens: Maximum number of tokens to generate.
                Defaults to 512.
            temperature: Sampling temperature (0.0 to 2.0).
                Higher values produce more random outputs.
                Defaults to 0.7.
            do_sample: Whether to use sampling. If False, uses greedy decoding.
                Defaults to True.
            top_k: Top-k sampling parameter. Defaults to 50.
            top_p: Nucleus sampling parameter. Defaults to 0.95.
            skip_special_tokens: Whether to skip special tokens in output.
                Defaults to True.
            **kwargs: Backend-specific parameters:

                For 'transformers':
                    - device: Device placement ('auto', 'cuda', 'cpu', 'mps')
                    - torch_dtype: Model precision (torch.float16, etc.)
                    - trust_remote_code: Trust remote code (default: False)

                For 'unsloth':
                    - max_seq_length: Maximum sequence length (default: 2048)
                    - load_in_4bit: Use 4-bit quantization (default: True)

                For 'vllm':
                    - tensor_parallel_size: GPUs for tensor parallelism (default: 1)
                    - gpu_memory_utilization: GPU memory fraction (default: 0.9)
                    - trust_remote_code: Trust remote code (default: False)

                For 'llamacpp':
                    - n_gpu_layers: Layers to offload to GPU (default: 0)
                    - n_ctx: Context window size (default: 2048)
                    - repo_id: HuggingFace repo for download
                    - filename: GGUF filename to download
                    - verbose: Enable verbose logging (default: False)
                    - seed: Random seed (default: -1)

        Returns:
            An initialized backend instance (subclass of BaseBackend).

        Raises:
            ValueError: If an unknown backend is specified.
            ImportError: If required dependencies are not installed.
            RuntimeError: If model loading fails.

        Example:
            >>> model = AutoInference.from_pretrained(
            ...     "gpt2",
            ...     backend="transformers",
            ...     max_new_tokens=100
            ... )
            >>> print(model("Hello, how are you?"))
        """
        # Validate backend
        if backend not in cls._BACKEND_REGISTRY:
            available = ", ".join(cls._BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Available backends: {available}"
            )

        # Create generation config
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=skip_special_tokens,
        )

        # Load backend class dynamically
        backend_class = cls._get_backend_class(backend)

        # Create and return backend instance
        return backend_class(
            model_name=model_name,
            config=config,
            **kwargs,
        )

    @classmethod
    def _get_backend_class(cls, backend: str) -> Type[BaseBackend]:
        """
        Dynamically import and return the backend class.

        Args:
            backend: Name of the backend to load.

        Returns:
            The backend class (not an instance).

        Raises:
            ImportError: If the backend module cannot be imported.
        """
        module_path = cls._MODULE_PATHS[backend]
        class_name = cls._BACKEND_REGISTRY[backend]

        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            # Provide helpful error message for missing dependencies
            dep_hints = {
                "transformers": "pip install transformers torch",
                "unsloth": "pip install unsloth",
                "vllm": "pip install vllm",
                "llamacpp": "pip install llama-cpp-python",
            }
            hint = dep_hints.get(backend, "")
            raise ImportError(
                f"Failed to load {backend} backend. "
                f"Ensure dependencies are installed: {hint}"
            ) from e

    @classmethod
    def available_backends(cls) -> list:
        """
        Get a list of available backend names.

        Returns:
            List of backend name strings.

        Example:
            >>> AutoInference.available_backends()
            ['transformers', 'unsloth', 'vllm', 'llamacpp']
        """
        return list(cls._BACKEND_REGISTRY.keys())

    @classmethod
    def get_backend_info(cls, backend: str) -> Dict[str, str]:
        """
        Get information about a specific backend.

        Args:
            backend: Name of the backend.

        Returns:
            Dictionary with backend information.

        Example:
            >>> info = AutoInference.get_backend_info("transformers")
            >>> print(info['description'])
        """
        info = {
            "transformers": {
                "name": "Transformers",
                "description": "HuggingFace Transformers for flexible inference",
                "requirements": "transformers, torch",
                "gpu_required": False,
                "best_for": "Development, research, compatibility",
            },
            "unsloth": {
                "name": "Unsloth",
                "description": "Memory-efficient inference with 4-bit quantization",
                "requirements": "unsloth",
                "gpu_required": True,
                "best_for": "Fine-tuned models, memory-constrained GPUs",
            },
            "vllm": {
                "name": "vLLM",
                "description": "High-throughput inference with PagedAttention",
                "requirements": "vllm",
                "gpu_required": True,
                "best_for": "Production servers, batch processing",
            },
            "llamacpp": {
                "name": "llama.cpp",
                "description": "CPU-optimized inference for GGUF models",
                "requirements": "llama-cpp-python",
                "gpu_required": False,
                "best_for": "Edge devices, CPU inference, laptops",
            },
        }
        if backend not in info:
            raise ValueError(f"Unknown backend: {backend}")
        return info[backend]

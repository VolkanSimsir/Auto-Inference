from typing import Any, Dict, List, Literal, Optional, Type

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
        - ``'transformers'``: HuggingFace Transformers — flexible, CPU/GPU, quantization
        - ``'unsloth'``: Unsloth — memory-efficient 4-bit fine-tuned models
        - ``'vllm'``: vLLM — high-throughput PagedAttention, production servers
        - ``'llamacpp'``: llama.cpp — GGUF models, CPU/edge deployment

    Example:
        Transformers backend (with 4-bit quantization)::

            from auto_inference import AutoInference

            model = AutoInference.from_pretrained(
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                backend="transformers",
                load_in_4bit=True,
                max_new_tokens=256,
            )
            print(model("What is machine learning?"))

    Example:
        Streaming generation::

            for chunk in model.generate_stream("Tell me a joke"):
                print(chunk, end="", flush=True)

    Example:
        Multi-turn chat::

            messages = [{"role": "user", "content": "Hello!"}]
            response = model.chat(messages, system_prompt="You are helpful.")

    Example:
        vLLM for high-throughput inference::

            model = AutoInference.from_pretrained(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                backend="vllm",
                tensor_parallel_size=2,
            )

    Example:
        CPU inference with llama.cpp::

            model = AutoInference.from_pretrained(
                model_name="/path/to/model.gguf",
                backend="llamacpp",
                n_gpu_layers=0,
                n_ctx=4096,
            )

    Note:
        The factory returns an instance of the appropriate backend class,
        not an ``AutoInference`` instance. This allows direct access to
        backend-specific features while maintaining a consistent API.
    """

    _BACKEND_REGISTRY: Dict[str, str] = {
        "transformers": "TransformersBackend",
        "unsloth": "UnslothBackend",
        "vllm": "VLLMBackend",
        "llamacpp": "LlamaCppBackend",
    }

    _MODULE_PATHS: Dict[str, str] = {
        "transformers": "auto_inference.backends.transformers_backend",
        "unsloth": "auto_inference.backends.unsloth_backend",
        "vllm": "auto_inference.backends.vllm_backend",
        "llamacpp": "auto_inference.backends.llamacpp_backend",
    }

    def __init__(self) -> None:
        raise RuntimeError(
            "AutoInference should not be instantiated directly. "
            "Use AutoInference.from_pretrained() instead."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        backend: BackendType = "transformers",
        # --- GenerationConfig params ---
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        skip_special_tokens: bool = True,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseBackend:
        """
        Create an inference backend from a pretrained model.

        Args:
            model_name: Model identifier or path.
                - HuggingFace model ID: ``'meta-llama/Meta-Llama-3-8B-Instruct'``
                - Local directory: ``'/path/to/model'``
                - GGUF file (llamacpp): ``'/path/to/model.gguf'``
            backend: Inference engine to use. Defaults to ``'transformers'``.
            max_new_tokens: Maximum tokens to generate. Defaults to 512.
            temperature: Sampling temperature 0–2. Defaults to 0.7.
            do_sample: Use sampling (True) or greedy decoding (False). Defaults to True.
            top_k: Top-k sampling parameter. Defaults to 50.
            top_p: Nucleus sampling threshold. Defaults to 0.95.
            repetition_penalty: Penalise repeated tokens (>= 1.0). Defaults to 1.0.
            skip_special_tokens: Strip special tokens from output. Defaults to True.
            stop_sequences: List of strings that stop generation early.
            **kwargs: Backend-specific parameters:

                *transformers*:
                    - ``device``: ``'auto'`` | ``'cuda'`` | ``'cpu'`` | ``'mps'``
                    - ``torch_dtype``: ``torch.float16`` / ``torch.bfloat16`` etc.
                    - ``trust_remote_code``: bool (default False)
                    - ``load_in_4bit``: bool — BitsAndBytes 4-bit quantization
                    - ``load_in_8bit``: bool — BitsAndBytes 8-bit quantization

                *unsloth*:
                    - ``max_seq_length``: int (default 2048)
                    - ``load_in_4bit``: bool (default True)

                *vllm*:
                    - ``tensor_parallel_size``: int (default 1)
                    - ``gpu_memory_utilization``: float (default 0.9)
                    - ``trust_remote_code``: bool

                *llamacpp*:
                    - ``n_gpu_layers``: int (default 0)
                    - ``n_ctx``: int (default 2048)
                    - ``repo_id``: str — HuggingFace repo to download from
                    - ``filename``: str — GGUF filename
                    - ``verbose``: bool (default False)

        Returns:
            An initialized backend instance (subclass of :class:`BaseBackend`).

        Raises:
            ValueError: If an unknown backend is specified.
            ImportError: If required dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        if backend not in cls._BACKEND_REGISTRY:
            available = ", ".join(cls._BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend: '{backend}'. Available: {available}"
            )

        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
            stop_sequences=stop_sequences or [],
        )

        backend_class = cls._get_backend_class(backend)
        return backend_class(model_name=model_name, config=config, **kwargs)

    @classmethod
    def _get_backend_class(cls, backend: str) -> Type[BaseBackend]:
        module_path = cls._MODULE_PATHS[backend]
        class_name = cls._BACKEND_REGISTRY[backend]

        dep_hints = {
            "transformers": "pip install 'auto-inference[transformers]'",
            "unsloth": "pip install 'auto-inference[unsloth]'",
            "vllm": "pip install 'auto-inference[vllm]'",
            "llamacpp": "pip install 'auto-inference[llamacpp]'",
        }

        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            hint = dep_hints.get(backend, "")
            raise ImportError(
                f"Failed to load '{backend}' backend. "
                f"Install its dependencies with: {hint}"
            ) from e

    @classmethod
    def available_backends(cls) -> List[str]:
        """Return a list of all supported backend names."""
        return list(cls._BACKEND_REGISTRY.keys())

    @classmethod
    def get_backend_info(cls, backend: str) -> Dict[str, Any]:
        """
        Return metadata about a backend.

        Args:
            backend: Backend name.

        Returns:
            Dict with keys: name, description, requirements, gpu_required, best_for.
        """
        info: Dict[str, Any] = {
            "transformers": {
                "name": "Transformers",
                "description": (
                    "HuggingFace Transformers — flexible CPU/GPU inference with "
                    "optional 4-bit/8-bit BitsAndBytes quantization, streaming, "
                    "and chat template support."
                ),
                "requirements": "transformers, accelerate, torch",
                "optional": "bitsandbytes (for quantization)",
                "gpu_required": False,
                "best_for": "Development, research, quantized inference",
            },
            "unsloth": {
                "name": "Unsloth",
                "description": (
                    "Memory-efficient inference optimised for fine-tuned models "
                    "with native 4-bit quantization."
                ),
                "requirements": "unsloth",
                "optional": "",
                "gpu_required": True,
                "best_for": "Fine-tuned models on memory-constrained GPUs",
            },
            "vllm": {
                "name": "vLLM",
                "description": (
                    "High-throughput inference engine using PagedAttention. "
                    "Supports continuous batching and tensor parallelism."
                ),
                "requirements": "vllm",
                "optional": "",
                "gpu_required": True,
                "best_for": "Production servers, batch workloads",
            },
            "llamacpp": {
                "name": "llama.cpp",
                "description": (
                    "GGUF model inference via llama.cpp bindings — runs on CPU "
                    "with optional GPU layer offloading."
                ),
                "requirements": "llama-cpp-python",
                "optional": "",
                "gpu_required": False,
                "best_for": "Edge devices, laptops, CPU-only inference",
            },
        }
        if backend not in info:
            raise ValueError(f"Unknown backend: '{backend}'")
        return info[backend]

from typing import Any, List, Optional, Union

from auto_inference.backends.base import BaseBackend, GenerationConfig


class VLLMBackend(BaseBackend):
    """
    Inference backend using vLLM for high-throughput serving.

    vLLM is designed for production-grade LLM serving with features like:
    - PagedAttention for efficient KV cache management
    - Continuous batching for high throughput
    - Optimized CUDA kernels

    This backend is ideal for:
    - Production API servers
    - Batch inference workloads
    - High-concurrency scenarios

    Attributes:
        model_name: HuggingFace model ID or local path.
        config: Generation configuration parameters.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        model: The loaded vLLM LLM instance.

    Example:
        Basic single-GPU usage:

        >>> from auto_inference.backends import VLLMBackend
        >>> from auto_inference.backends.base import GenerationConfig
        >>>
        >>> backend = VLLMBackend(
        ...     model_name="meta-llama/Llama-2-7b-chat-hf",
        ...     config=GenerationConfig(max_new_tokens=256)
        ... )
        >>> response = backend("What is artificial intelligence?")
        >>> print(response)

    Example:
        Multi-GPU tensor parallel inference:

        >>> backend = VLLMBackend(
        ...     model_name="meta-llama/Llama-2-70b-chat-hf",
        ...     config=GenerationConfig(
        ...         max_new_tokens=512,
        ...         temperature=0.7,
        ...         top_p=0.9
        ...     ),
        ...     tensor_parallel_size=4,
        ...     gpu_memory_utilization=0.9
        ... )
        >>> response = backend("Explain quantum computing")

    Example:
        Batch inference for multiple prompts:

        >>> backend = VLLMBackend("mistralai/Mistral-7B-v0.1")
        >>> prompts = [
        ...     "What is Python?",
        ...     "Explain machine learning",
        ...     "What is a neural network?"
        ... ]
        >>> responses = backend.generate_batch(prompts)
        >>> for prompt, response in zip(prompts, responses):
        ...     print(f"Q: {prompt}")
        ...     print(f"A: {response}\\n")

    Note:
        - Requires `vllm` package to be installed.
        - CUDA-capable GPU is required.
        - For optimal performance, ensure GPU memory utilization is tuned.
        - Tensor parallelism requires multiple GPUs on the same node.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the vLLM backend.

        Args:
            model_name: HuggingFace model ID or local path to the model.
            config: Generation configuration. Uses defaults if not provided.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
                Set to > 1 for large models that don't fit on a single GPU.
                Defaults to 1 (single GPU).
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
                Higher values allow larger batch sizes but may cause OOM errors.
                Defaults to 0.9.
            trust_remote_code: Whether to trust remote code for custom models.
                Defaults to False for security.
            **kwargs: Additional arguments passed to vLLM's LLM class.

        Raises:
            ImportError: If vllm is not installed.
            RuntimeError: If CUDA is not available.
            ValueError: If tensor_parallel_size exceeds available GPUs.
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self._sampling_params = None
        super().__init__(model_name, config, **kwargs)

    def _load_model(self) -> None:
        """
        Load the model using vLLM.

        This method initializes the vLLM LLM engine and configures
        sampling parameters based on the generation config.

        Raises:
            ImportError: If vllm package is not installed.
            RuntimeError: If model loading fails.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm package is required for VLLMBackend. "
                "Install it with: pip install vllm"
            ) from e

        # Filter out unsupported kwargs (e.g., 'device' which vLLM manages internally)
        unsupported_keys = {"device", "torch_dtype"}
        filtered_kwargs = {k: v for k, v in self._kwargs.items() if k not in unsupported_keys}

        # Initialize the LLM engine
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            **filtered_kwargs,
        )

        # Configure sampling parameters
        self._sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else 0.0,
            top_k=self.config.top_k if self.config.do_sample else -1,
            top_p=self.config.top_p if self.config.do_sample else 1.0,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: The input text to generate a response for.

        Returns:
            Generated text response.

        Raises:
            ValueError: If prompt is empty or None.
            RuntimeError: If generation fails.

        Example:
            >>> backend = VLLMBackend("gpt2")
            >>> response = backend.generate("The future of AI is")
            >>> print(response)
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        outputs = self.model.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[Any] = None,
    ) -> List[str]:
        """
        Generate text for multiple prompts in a batch.

        This method leverages vLLM's continuous batching for efficient
        processing of multiple prompts simultaneously.

        Args:
            prompts: List of input prompts to process.
            sampling_params: Optional custom SamplingParams. If None,
                uses the default parameters from config.

        Returns:
            List of generated text responses in the same order as inputs.

        Raises:
            ValueError: If prompts list is empty.
            RuntimeError: If batch generation fails.

        Example:
            >>> backend = VLLMBackend("mistralai/Mistral-7B-v0.1")
            >>> prompts = ["Hello", "What is AI?", "Explain Python"]
            >>> responses = backend.generate_batch(prompts)
            >>> for r in responses:
            ...     print(r)
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        params = sampling_params or self._sampling_params
        outputs = self.model.generate(prompts, params)

        return [output.outputs[0].text.strip() for output in outputs]

    def update_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        """
        Update sampling parameters without reloading the model.

        This is useful for dynamically adjusting generation parameters
        between requests without the overhead of reinitializing.

        Args:
            max_tokens: New max tokens value. If None, keeps current.
            temperature: New temperature value. If None, keeps current.
            top_k: New top_k value. If None, keeps current.
            top_p: New top_p value. If None, keeps current.

        Example:
            >>> backend = VLLMBackend("gpt2")
            >>> backend.update_sampling_params(temperature=0.9, max_tokens=100)
            >>> response = backend("Creative story:")
        """
        from vllm import SamplingParams

        self._sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config.max_new_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_k=top_k if top_k is not None else self.config.top_k,
            top_p=top_p if top_p is not None else self.config.top_p,
        )

import os
from typing import Any, Optional

from auto_inference.backends.base import BaseBackend, GenerationConfig


class LlamaCppBackend(BaseBackend):
    """
    Inference backend using llama-cpp-python for CPU/GPU inference.

    This backend is optimized for:
    - CPU inference with AVX/AVX2/AVX-512 acceleration
    - Quantized GGUF models (Q4, Q5, Q8, etc.)
    - Edge devices and consumer hardware
    - Optional GPU offloading for hybrid CPU/GPU inference

    Supports loading models from:
    - Local GGUF files
    - HuggingFace Hub (automatic download)

    Attributes:
        model_name: Path to local GGUF file or model identifier.
        config: Generation configuration parameters.
        n_gpu_layers: Number of layers to offload to GPU.
        n_ctx: Context window size.
        model: The loaded Llama instance.

    Example:
        Loading a local GGUF file:

        >>> from auto_inference.backends import LlamaCppBackend
        >>> from auto_inference.backends.base import GenerationConfig
        >>>
        >>> backend = LlamaCppBackend(
        ...     model_name="/path/to/model.gguf",
        ...     config=GenerationConfig(max_new_tokens=256),
        ...     n_gpu_layers=35  # Offload 35 layers to GPU
        ... )
        >>> response = backend("What is machine learning?")
        >>> print(response)

    Example:
        Downloading from HuggingFace Hub:

        >>> backend = LlamaCppBackend(
        ...     model_name="llama-7b",  # Identifier (not used if repo_id provided)
        ...     config=GenerationConfig(max_new_tokens=512),
        ...     repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
        ...     filename="llama-2-7b-chat.Q4_K_M.gguf",
        ...     n_gpu_layers=-1  # Use all available GPU layers
        ... )
        >>> response = backend("Explain quantum computing")

    Example:
        Pure CPU inference (no GPU):

        >>> backend = LlamaCppBackend(
        ...     model_name="/path/to/model.gguf",
        ...     n_gpu_layers=0,  # CPU only
        ...     n_ctx=4096
        ... )
        >>> response = backend("Hello, how are you?")

    Note:
        - Requires `llama-cpp-python` package to be installed.
        - For GPU support, install with: pip install llama-cpp-python[cuda]
        - GGUF is the recommended format for llama.cpp models.
        - Quantized models (Q4_K_M, Q5_K_M) offer good quality/size tradeoffs.

    Warning:
        Large context windows (n_ctx) significantly increase memory usage.
        Adjust based on available RAM/VRAM.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        verbose: bool = False,
        seed: int = -1,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Llama.cpp backend.

        Args:
            model_name: Path to a local GGUF file or a model identifier.
                If repo_id is provided, this is used only as an identifier.
            config: Generation configuration. Uses defaults if not provided.
            n_gpu_layers: Number of model layers to offload to GPU.
                - 0: Pure CPU inference
                - -1: Offload all layers to GPU
                - N: Offload first N layers to GPU
                Defaults to 0 (CPU only).
            n_ctx: Context window size in tokens. Larger values allow longer
                conversations but require more memory. Defaults to 2048.
            verbose: Whether to print detailed loading and inference logs.
                Defaults to False.
            seed: Random seed for reproducibility. -1 for random seed.
                Defaults to -1.
            repo_id: HuggingFace repository ID for downloading models.
                Example: "TheBloke/Llama-2-7B-Chat-GGUF"
            filename: Specific GGUF filename to download from the repository.
                Example: "llama-2-7b-chat.Q4_K_M.gguf"
            **kwargs: Additional arguments passed to llama_cpp.Llama.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If local model file doesn't exist.
            ValueError: If repo_id is provided without filename.
        """
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.seed = seed
        self.repo_id = repo_id
        self.filename = filename
        self._model_path: Optional[str] = None
        super().__init__(model_name, config, **kwargs)

    def _download_from_huggingface(self) -> str:
        """
        Download a GGUF model from HuggingFace Hub.

        Returns:
            Path to the downloaded model file.

        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If filename is not specified.
            Exception: If download fails.
        """
        if not self.filename:
            raise ValueError(
                "filename must be specified when downloading from HuggingFace. "
                "Example: filename='model.Q4_K_M.gguf'"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Install it with: pip install huggingface_hub"
            ) from e

        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )

        return model_path

    def _load_model(self) -> None:
        """
        Load the GGUF model using llama-cpp-python.

        This method handles:
        1. Downloading from HuggingFace if repo_id is specified
        2. Loading local GGUF files
        3. Configuring GPU layers and context size

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install it with: pip install llama-cpp-python\n"
                "For GPU support: pip install llama-cpp-python[cuda]"
            ) from e

        # Determine model path
        if self.repo_id:
            self._model_path = self._download_from_huggingface()
        else:
            self._model_path = self.model_name
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self._model_path}. "
                    "Provide a valid path or use repo_id/filename for HuggingFace."
                )

        # Load the model
        self.model = Llama(
            model_path=self._model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
            seed=self.seed,
            **self._kwargs,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from the given prompt.

        Args:
            prompt: The input text to generate a response for.

        Returns:
            Generated text response.

        Raises:
            ValueError: If prompt is empty or None.
            RuntimeError: If generation fails.

        Example:
            >>> backend = LlamaCppBackend("/path/to/model.gguf")
            >>> response = backend.generate("The capital of France is")
            >>> print(response)
            "Paris, which is known for..."
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Generate using llama.cpp
        output = self.model(
            prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            echo=False,  # Don't include prompt in output
        )

        # Extract generated text
        if isinstance(output, dict) and "choices" in output:
            return output["choices"][0]["text"].strip()
        return str(output).strip()

    def generate_with_grammar(
        self,
        prompt: str,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Generate text with grammar constraints.

        This method allows generating structured outputs (JSON, etc.)
        by constraining the output to follow a specified grammar.

        Args:
            prompt: The input text to generate a response for.
            grammar: A llama_cpp.LlamaGrammar instance defining the
                output structure. See llama-cpp-python documentation.

        Returns:
            Generated text following the grammar constraints.

        Example:
            >>> from llama_cpp import LlamaGrammar
            >>> json_grammar = LlamaGrammar.from_string(...)
            >>> backend = LlamaCppBackend("/path/to/model.gguf")
            >>> response = backend.generate_with_grammar(
            ...     "Extract the name and age:",
            ...     grammar=json_grammar
            ... )
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        output = self.model(
            prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            grammar=grammar,
            echo=False,
        )

        if isinstance(output, dict) and "choices" in output:
            return output["choices"][0]["text"].strip()
        return str(output).strip()

    @property
    def model_path(self) -> Optional[str]:
        """Get the path to the loaded model file."""
        return self._model_path

    @property
    def context_size(self) -> int:
        """Get the configured context window size."""
        return self.n_ctx

    @property
    def gpu_layers(self) -> int:
        """Get the number of layers offloaded to GPU."""
        return self.n_gpu_layers

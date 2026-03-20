from threading import Thread
from typing import Any, Iterator, List, Literal, Optional

import torch

from auto_inference.backends.base import BaseBackend, GenerationConfig, Message


class TransformersBackend(BaseBackend):
    """
    Inference backend using HuggingFace Transformers.

    Supports automatic device placement, mixed precision, BitsAndBytes quantization
    (4-bit / 8-bit), streaming generation, chat templates, and batch inference.

    Args:
        model_name: HuggingFace model ID or local path.
        config: Generation configuration.
        device: Device for inference ('auto', 'cuda', 'cpu', 'mps').
        torch_dtype: Model precision. Auto-selected if None.
        trust_remote_code: Allow executing remote model code.
        load_in_4bit: Load model in 4-bit quantization via BitsAndBytes.
        load_in_8bit: Load model in 8-bit quantization via BitsAndBytes.
        bnb_4bit_quant_type: 4-bit quantization type ('nf4' or 'fp4').
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantized layers.
        **kwargs: Passed to `AutoModelForCausalLM.from_pretrained`.

    Example:
        Basic usage::

            from auto_inference.backends import TransformersBackend

            backend = TransformersBackend("meta-llama/Meta-Llama-3-8B-Instruct")
            print(backend("Explain transformers in one sentence."))

    Example:
        4-bit quantization::

            backend = TransformersBackend(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                load_in_4bit=True,
            )

    Example:
        Multi-turn chat::

            messages = [{"role": "user", "content": "What is Python?"}]
            print(backend.chat(messages, system_prompt="You are a helpful assistant."))

    Example:
        Streaming::

            for chunk in backend.generate_stream("Once upon a time"):
                print(chunk, end="", flush=True)

    Note:
        Requires ``transformers``, ``torch``, and ``accelerate``.
        BitsAndBytes quantization additionally requires ``bitsandbytes``.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[GenerationConfig] = None,
        device: Literal["auto", "cuda", "cpu", "mps"] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> None:
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit cannot both be True.")

        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        super().__init__(model_name, config, **kwargs)

    def _load_model(self) -> None:
        """Load model and tokenizer from HuggingFace or a local path."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "transformers package is required for TransformersBackend. "
                "Install it with: pip install transformers accelerate"
            ) from e

        # Determine dtype
        if self.torch_dtype is None:
            if self.device == "cpu":
                self.torch_dtype = torch.float32
            else:
                self.torch_dtype = torch.bfloat16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build model kwargs
        model_kwargs: dict = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            **self._kwargs,
        }

        # BitsAndBytes quantization config
        if self.load_in_4bit or self.load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for quantized loading. "
                    "Install it with: pip install bitsandbytes"
                ) from e

            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model_kwargs["quantization_config"] = bnb_config
            # Quantized models must use device_map
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype
            model_kwargs["device_map"] = "auto" if self.device == "auto" else None

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Move to explicit device when not using device_map
        if self.device not in ("auto", "cpu") and not (self.load_in_4bit or self.load_in_8bit):
            self.model = self.model.to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt (single call, full response).

        Args:
            prompt: Input text.

        Returns:
            Generated text (without the original prompt).
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        inputs = self._tokenize(prompt)

        generate_kwargs = self._build_generate_kwargs()

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=self.config.skip_special_tokens,
        ).strip()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Stream generated tokens incrementally using TextIteratorStreamer.

        Yields text chunks as they are produced so the caller can display
        output in real time without waiting for the full response.

        Args:
            prompt: Input text.

        Yields:
            Text chunks (may be single tokens or small groups).

        Example:
            >>> for chunk in backend.generate_stream("Hello!"):
            ...     print(chunk, end="", flush=True)
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        try:
            from transformers import TextIteratorStreamer
        except ImportError as e:
            raise ImportError(
                "TextIteratorStreamer requires transformers >= 4.35.0"
            ) from e

        inputs = self._tokenize(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=self.config.skip_special_tokens,
        )

        generate_kwargs = {**inputs, **self._build_generate_kwargs(), "streamer": streamer}

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for a list of prompts in a single batched forward pass.

        Args:
            prompts: List of input prompts.

        Returns:
            List of generated responses, same order as input.

        Example:
            >>> responses = backend.generate_batch(["Hello", "Hi there", "Good morning"])
        """
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = self._move_to_device(inputs)

        generate_kwargs = self._build_generate_kwargs()

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        results = []
        for i, output in enumerate(outputs):
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[input_length:]
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=self.config.skip_special_tokens,
            )
            results.append(text.strip())

        return results

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for a multi-turn conversation using the model's
        chat template (via ``tokenizer.apply_chat_template``).

        If the tokenizer does not have a chat template, falls back to a
        plain text concatenation of messages.

        Args:
            messages: Conversation history as a list of dicts with
                'role' and 'content' keys. Roles: 'user', 'assistant', 'system'.
            system_prompt: Optional system-level instruction prepended as a
                'system' role message before the user conversation.

        Returns:
            The assistant's response text.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is 2 + 2?"},
            ... ]
            >>> backend.chat(messages, system_prompt="Answer concisely.")
            '4'
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for models without a chat template
            parts = []
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

        return self.generate(prompt)

    def chat_stream(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream a chat response token by token.

        Combines chat template formatting with streaming generation.

        Args:
            messages: Conversation history.
            system_prompt: Optional system prompt.

        Yields:
            Text chunks as they are generated.

        Example:
            >>> for chunk in backend.chat_stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            parts = []
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

        yield from self.generate_stream(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return self._move_to_device(inputs)

    def _move_to_device(self, inputs: dict) -> dict:
        if self.device == "auto" or self.load_in_4bit or self.load_in_8bit:
            return {k: v.to(self.model.device) for k, v in inputs.items()}
        if self.device != "cpu":
            return {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _build_generate_kwargs(self) -> dict:
        """Build kwargs for model.generate() from the current config."""
        kwargs: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }

        if self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_k"] = self.config.top_k
            kwargs["top_p"] = self.config.top_p

        if self.config.stop_sequences:
            stop_ids = [
                self.tokenizer.encode(s, add_special_tokens=False)
                for s in self.config.stop_sequences
            ]
            # Only use single-token stops for simplicity
            single_token_stops = [ids[0] for ids in stop_ids if len(ids) == 1]
            if single_token_stops:
                existing = kwargs.get("eos_token_id", [])
                if isinstance(existing, int):
                    existing = [existing]
                kwargs["eos_token_id"] = list(set(existing + single_token_stops))

        return kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device_info(self) -> str:
        """Human-readable description of the device configuration."""
        if self.model is None:
            return "Model not loaded"
        if hasattr(self.model, "hf_device_map"):
            return f"Multi-device: {self.model.hf_device_map}"
        return f"Device: {next(self.model.parameters()).device}"

    @property
    def quantization_info(self) -> str:
        """Describe the active quantization mode."""
        if self.load_in_4bit:
            return f"4-bit ({self.bnb_4bit_quant_type})"
        if self.load_in_8bit:
            return "8-bit"
        return f"None ({self.torch_dtype})"

    def __repr__(self) -> str:
        return (
            f"TransformersBackend("
            f"model='{self.model_name}', "
            f"device='{self.device}', "
            f"quantization={self.quantization_info}, "
            f"max_new_tokens={self.config.max_new_tokens})"
        )

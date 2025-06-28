from typing import Literal, Optional
import importlib
import logging

LIB_TYPE_TO_MODEL_CLASS_NAME = {
    "transformers": "TransformersInference",
    "unsloth": "UnslothInference",
    "vllm": "VLLMInference",
    "llamacpp": "LlamaCppInference",
    "sglang": "SglangInference",
}

class AutoInference:
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        lib_type: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        skip_special_tokens: bool = True,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        verbose: bool = False,
        seed: int = 1337,
        echo: bool = True,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        device: str = "auto",
        max_seq_length: int = 2048,  
        load_in_4bit: bool = True,
        top_k: int = 50,
        top_p:float = 0.95,
    ):
        lib_class_name = LIB_TYPE_TO_MODEL_CLASS_NAME[lib_type]
        
        lib = __import__(f"autoinference.lib.{lib_type}", fromlist=[lib_class_name])
        InfLib = getattr(lib, lib_class_name)
        
        return InfLib(
            model_name=model_name,
            lib_type=lib_type,
            device=device,  # Pass device parameter
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            skip_special_tokens=skip_special_tokens,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            seed=seed,
            echo=echo,
            repo_id=repo_id,
            filename=filename,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            top_k = top_k,
            top_p = top_p
        )
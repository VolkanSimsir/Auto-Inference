from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import logging
import os

class LlamaCppInference:
    def __init__(
        self,
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
        repo_id: str = None,
        filename: str = None,
        **kwargs
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.seed = seed
        self.echo = echo
        self.repo_id = repo_id or model_name
        self.filename = filename
        
        self.model = None
        self._load_model()

    def _load_model(self):
        logging.info("Loading LlamaCpp model...")
        
        if self.filename:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir="./models"
            )
            model_path = os.path.join("./models", self.filename)
        else:

            model_path = self.model_name
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=self.n_gpu_layers,
            seed=self.seed,
            n_ctx=self.n_ctx,
            verbose=self.verbose
        )
        
        logging.info("Model loaded successfully.")

    def __call__(self, prompt: str) -> str:
        logging.info("Generating response...")
        response = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            echo=self.echo,
        )
        
        result = response['choices'][0]['text']
        logging.info("Response generated successfully.")
        return result
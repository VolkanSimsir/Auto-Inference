from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

class TransformersInference:
    def __init__(
        self,
        model_name: str,
        lib_type: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        skip_special_tokens: bool = True,
        **kwargs  
    ):
        self.device = device
        self.model_name = model_name
        self.lib_type = lib_type
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.skip_special_tokens = skip_special_tokens
        self._load_model()
        
    def _load_model(self):
        logging.info("Loading Transformers model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logging.info("Model loaded successfully.")
        
    def __call__(self, prompt: str) -> str:
        logging.info("Generating response...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        logging.info("Response generated successfully.")
        return response[0] if response else ""
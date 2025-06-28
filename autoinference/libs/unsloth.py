from transformers import TextStreamer
import torch
import logging

class UnslothInference:
    def __init__(
        self,
        model_name: str,
        lib_type: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        skip_special_tokens: bool = True,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.skip_special_tokens = skip_special_tokens
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        
        self._load_model()

    def _load_model(self):
        from unsloth import FastLanguageModel
        
        logging.info("Loading Unsloth model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        
        FastLanguageModel.for_inference(self.model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logging.info("Model loaded successfully.")

    def __call__(self, prompt: str) -> str:
        logging.info("Generating response...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=self.skip_special_tokens)
        logging.info("Response generated successfully.")
        return response

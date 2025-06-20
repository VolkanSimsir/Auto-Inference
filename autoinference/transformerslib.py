
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import ModelConfig 
import torch
import logging

class TransformersInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def load_model(self):
        logging.info("Transformers model loading...")
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, torch_dtype=torch.float16).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        logging.info("Model loaded successfully.")
        return model, tokenizer

    def generate(self, prompt):
        logging.info(f"Generating response...")
        model, tokenizer = self.load_model()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=True)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logging.info("Response generated successfully.")
        return response
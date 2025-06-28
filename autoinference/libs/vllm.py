from vllm import LLM, SamplingParams
import logging

class VLLMInference:
    def __init__(
        self,
        model_name: str,
        lib_type: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        skip_special_tokens: bool = True,
        top_p:float = 0.95,
        top_k:int = 50,
        **kwargs
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.skip_special_tokens = skip_special_tokens
        
        self._load_model()

    def _load_model(self):
        logging.info("Loading vLLM model...")
        self.model = LLM(model=self.model_name)
        logging.info("Model loaded successfully.")

    def __call__(self, prompt: str) -> str:
        logging.info("Generating response...")
        
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p = self.top_p,
            top_k = self.top_k
        )
        outputs = self.model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text
        logging.info("Response generated successfully.")
        return response
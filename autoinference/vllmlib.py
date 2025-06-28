from .config import ModelConfig
from vllm import LLM, SamplingParams

class VLLMInference:
    """
    A class for running inference with vLLM.
    Args:
        config (ModelConfig): The configuration object for the model.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def load_model(self):
        """Loads the vLLM model."""
        self.model = LLM(model=self.config.model_name)
        print("vLLM model successfully loaded")
        return self.model

    def generate(self, prompt: str):
        if self.model is None:
            self.load_model()
        
        sampling_params = SamplingParams(max_tokens=self.config.max_new_tokens)
        
        outputs = self.model.generate(prompt, sampling_params)
        
        return outputs[0].outputs[0].text
from .config import ModelConfig
from .transformerslib import TransformersInference
from .unslothlib import UnslothInference
from .vllmlib import VLLMInference



class AutoInference:
    """
    A factory class for creating and using different inference engines.

    This class selects the appropriate inference engine based on the specified
    `model_type` and provides a simple interface for text generation.

    Args:
        model_name (str): The name of the model to load.
        model_type (str): The type of inference engine to use. 
                          Supported values are "transformers", "unsloth", and "vllm".

    """
    def __init__(self, model_name: str, model_type: str, load_in_4bit: bool):
        self.config = None
        self.engine = None
        self.model_name = model_name
        self.model_type = model_type
        
        

    def load_model(self):
        """
        Loads the inference engine based on the model type specified during initialization.
        
        It initializes a `ModelConfig` and then instantiates the correct
        inference engine (`TransformersInference`, `UnslothInference`, or `VLLMInference`).


        Returns:
            object: An instance of the loaded inference engine.
        """
        self.config = ModelConfig(
            model_name=self.model_name,
            model_type=self.model_type,
        )

        if self.config.model_type == "transformers":
            engine = TransformersInference(self.config)
        elif self.config.model_type == "unsloth":
            engine = UnslothInference(self.config)
        elif self.config.model_type == "vllm":
            engine = VLLMInference(self.config)

        self.engine = engine
        return self.engine

    def __call__(self, prompt: str) -> str:
        return self.load_model().generate(prompt)
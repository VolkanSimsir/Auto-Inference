from .config import ModelConfig
from .transformerslib import TransformersInference

class AutoInf:
    def __init__(self, model_name: str, model_type: str):
        self.config = None
        self.engine = None
        self.model_name = model_name
        self.model_type = model_type

    def load_model(self):
        self.config = ModelConfig(
            model_name=self.model_name,
            model_type=self.model_type,
        )
        if self.config.model_type == "transformers":
            engine = TransformersInference(self.config)

        self.engine = engine
        return self.engine

    def __call__(self, prompt: str) -> str:
        return self.load_model().generate(prompt)



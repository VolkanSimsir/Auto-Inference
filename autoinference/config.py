from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    max_new_tokens: int = 128
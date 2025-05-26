<div align="center">
<h2>
    AutoInference: Multiple inference options in a single library
</h2>
<img width="500" alt="teaser" src="assets\logo.jpeg">
</div>

## 🛠️ Installation

```bash
uv venv --python 3.12
.venv/bin/activate # for windows
source .venv/bin/activate # for linux
uv pip install -r requirements.txt
```

## Usage

```python
from autoinference.transformers.inference import llm_inference

llm_inference(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        prompt = "How are you?",
        max_new_tokens=100,
        temperature = 0.7
)
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

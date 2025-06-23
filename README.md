
<div align="center">
<h2>
    AutoInference: Multiple inference options in a single library
</h2>
<img width="350" alt="teaser" src="assets\logo.jpeg">
</div>



## What is AutoInference?
Auto-Inference is a Python library that provides a unified interface for model inference using several popular backends, including Hugging Face's Transformers and Unsloth.


## 🛠️ Installation

```bash
uv venv --python 3.12
.venv/bin/activate # for windows
source .venv/bin/activate # for linux
uv pip install -r requirements.txt
```

## Usage

🤗 Transformers library
```python
from autoinference import AutoInference

model = AutoInference(
        model_name="VolkanSimsir/LLaMA-3-8B-GRPO-math-tr", 
        model_type="transformers",
        
)

prompt = "How are you?"
output = model(prompt)
print(output)
```

🦥 Unsloth library
```python
from autoinference import AutoInference

model = AutoInference(
        model_name="unsloth/gemma-3-1b-it", 
        model_type="unsloth",
        
)

prompt = "How are you?"
output = model(prompt)
print(output)
```

## Supported Libraries

| Supported Libraries |  Status   
|----------------------|---|
| Transformers                 | ✅ 
| Unsloth                 | ✅
| vLLM              | x 
| Llama.cpp         | x



## 📜 License

This project is licensed under the terms of the Apache License 2.0.
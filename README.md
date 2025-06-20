
<div align="center">
<h2>
    AutoInference: Multiple inference options in a single library
</h2>
<img width="350" alt="teaser" src="assets\logo.jpeg">
</div>



## What is AutoInference?
Auto-Inference is a Python library that can infer models using Hugging Face's Transformers library. It currently only supports Transformers-based models. Unsloth and VLLM support are coming soon.


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
from autoinference import AutoInf

model = AutoInf(
        model_name="VolkanSimsir/LLaMA-3-8B-GRPO-math-tr", 
        model_type="transformers")

prompt = "How are you?"
output = model(prompt)
print(output)
```
## Supported Libraries

| Supported Libraries |  Status   
|----------------------|---|
| Transformers                 | ✅ 
| Unsloth                 | x
| vLLM              | x 

 

## 📜 License

This project is licensed under the terms of the Apache License 2.0.
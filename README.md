<div align="center">
<h2>
    AutoInference: Multiple inference options in a single library
</h2>
<img width="500" alt="teaser" src="assets\logo.jpeg">
</div>

## Latest News
* 01/06/2025: Added GPTQ and BitsAndBytes quantization techniques.

## What is AutoInference?
AutoInference is a Python library that supports large language models (LLM) and multiple inference backends (transformers) and quantization methods (GPTQ, AWQ, bitsandbytes). It provides model loading, optimization and inference with a single line.

## Quantization Support 
| Quantization Feature |  GPTQModel | Transformers | vLLM  | SGLang |
|----------------------|---|---|---|---|
| GPTQ                 | ✅ | ✅ | ✅ | ✅ |                        
| BitsAndBytes              | ✅ | ✅ | ✅ | ✅ |          
  


## 🔑 Features
* 🤗 HF Transformers Inference support
* ✨ Quantization techniques support


## 🛠️ Installation

```bash
uv venv --python 3.12
.venv/bin/activate # for windows
source .venv/bin/activate # for linux
uv pip install -r requirements.txt
```

## 🚀 Usage

### 🤗 Transformers Inference
```python
from autoinference.transformers import InfTransformers

prompt = "How are you?"
InfTransformers(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        prompt = prompt,
        max_new_tokens=100,
        temperature = 0.7
)
```
### Quantization Techniques
### 🔢 GPTQ Inference
```python
from autoinference import AutoInference

gptq_inf = AutoInference(
    model_name = "ModelName/",
    quant_method = "gptq",
    quant_config = {
            "bits": 4,
            "group_size": 128
            }
outputs = gptq_inf.generate("How are you?")
)
```
### ⚡ BitsAndBytes Inference
```python
from autoinference import AutoInference

bnb_inf = AutoInference(
    model_name = "ModelName/",
    quant_method = "BitsAndBytes",
    quant_config = {
            "load_in_4bit" = True,
            }
outputs = bnb_inf.generate("How are you?")
)
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

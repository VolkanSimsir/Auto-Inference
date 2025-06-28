<div align="center">
<h2>
    AutoInference: Multiple inference options in a single library
</h2>
<img width="350" alt="teaser" src="assets\logo.jpeg">
</div>



## What is AutoInference?
Auto-Inference is a Python library that provides a unified interface for model inference using several popular backends, including Hugging Face's Transformers, Unsloth, vLLM, and llama.cpp-python.Quantization support will be coming soon.



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

transformers_model = AutoInference.from_pretrained(
        model_name="VolkanSimsir/LLaMA-3-8B-GRPO-math-tr",
        lib_type="transformers",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        device= "auto"
)
response = transformers_model("Hello, how are you today?")
print(f"Transformers Response: {response}")

```

🦥 Unsloth library
```python
from autoinference import AutoInference

unsloth_model = AutoInference.from_pretrained(
        model_name="unsloth/gemma-3-1b-it", 
        lib_type="unsloth",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        max_seq_length=2048,     
        load_in_4bit=True,    
)

response = unsloth_model("What is artificial intelligence?")
print(f"Unsloth Response: {response}")

```

⚡️ vLLM library
```python
from autoinference import AutoInference

vllm_model = AutoInference.from_pretrained(
        model_name="VolkanSimsir/LLaMA-3-8B-GRPO-math-tr",
        lib_type="vllm",
        max_new_tokens=150,
        do_sample=True,
        temperature=0.9,
        #top_p=0.95,
        #top_k=50,
)

response = vllm_model("What is artificial intelligence?")
print(f"VLLM Response: {response}")
```
✨ Llama_cpp library
```python
from autoinference import AutoInference

llamacpp_model = AutoInference.from_pretrained(
        model_name="local_model.gguf",  # Local model file
        lib_type="llamacpp",
        max_new_tokens=100,
        temperature=0.6,
        n_gpu_layers=-1,            
        #n_ctx=4096,  

        # # To download from Hugging Face:
        #repo_id="unsloth/gemma-3-1b-it-GGUF",
        #filename="gemma-3-1b-it-Q4_K_M.gguf"
)

response = llamacpp_model("What is artificial intelligence?")
print(f"LlamaCpp Response: {response}")
```

## Supported Libraries

| Supported Libraries |  Status   
|----------------------|---|
| Transformers                 | ✅ 
| Unsloth                 | ✅
| vLLM              | ✅ 
| Llama.cpp         | ✅



## 📜 License

This project is licensed under the terms of the Apache License 2.0.
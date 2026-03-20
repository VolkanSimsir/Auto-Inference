<div align="center">
<h2>Auto-Inference: Unified LLM Inference for Multiple Backends</h2>
<img width="350" alt="logo" src="assets/logo.jpeg">
<br/><br/>

[![PyPI version](https://img.shields.io/pypi/v/auto-inference)](https://pypi.org/project/auto-inference/)
[![Python](https://img.shields.io/pypi/pyversions/auto-inference)](https://pypi.org/project/auto-inference/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

**Auto-Inference** provides a single, consistent Python API for running LLM inference across four popular backends — HuggingFace Transformers, vLLM, Unsloth, and llama.cpp — so you can swap engines with a one-word change.

## Features

| Feature | Transformers | vLLM | Unsloth | llama.cpp |
|---|:---:|:---:|:---:|:---:|
| Text generation | ✅ | ✅ | ✅ | ✅ |
| Streaming output | ✅ | — | — | — |
| Chat templates | ✅ | — | — | — |
| Batch generation | ✅ | ✅ | — | — |
| 4-bit quantization | ✅ BnB | — | ✅ native | ✅ GGUF |
| 8-bit quantization | ✅ BnB | — | — | — |
| CPU inference | ✅ | — | — | ✅ |
| Multi-GPU | ✅ | ✅ | — | — |

---

## Installation

```bash
# Create virtual environment (recommended)
uv venv --python 3.12 && source .venv/bin/activate

# Core + Transformers backend (most users start here)
pip install "auto-inference[transformers]"

# With BitsAndBytes quantization support
pip install "auto-inference[transformers,quantization]"

# vLLM (GPU required)
pip install "auto-inference[vllm]"

# llama.cpp (CPU / GGUF)
pip install "auto-inference[llamacpp]"

# Everything
pip install "auto-inference[all]"
```

---

## Quick Start

### Single prompt

```python
from auto_inference import AutoInference

model = AutoInference.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    backend="transformers",
    max_new_tokens=256,
    temperature=0.7,
)

print(model("What is quantum computing?"))
```

### Chat (multi-turn, with system prompt)

```python
messages = [
    {"role": "user", "content": "What is Python?"},
]

response = model.chat(
    messages,
    system_prompt="You are a concise technical assistant.",
)
print(response)

# Continue the conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "What about Rust?"})
print(model.chat(messages))
```

> Chat uses the model's own chat template via `tokenizer.apply_chat_template` when available — no manual formatting needed.

### Streaming

```python
for chunk in model.generate_stream("Tell me a short story about a robot:"):
    print(chunk, end="", flush=True)
print()

# Streaming chat
for chunk in model.chat_stream(messages, system_prompt="Be creative."):
    print(chunk, end="", flush=True)
```

### Batch generation

```python
prompts = [
    "Translate to French: Hello, how are you?",
    "Translate to German: Good morning!",
    "Translate to Spanish: Thank you very much.",
]

responses = model.generate_batch(prompts)
for p, r in zip(prompts, responses):
    print(f">> {r}")
```

---

## Backends

### 🤗 Transformers

```python
model = AutoInference.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    backend="transformers",
    device="auto",          # 'auto' | 'cuda' | 'cpu' | 'mps'
    max_new_tokens=512,
    temperature=0.7,
    repetition_penalty=1.1,
)
```

**4-bit quantization** (requires `bitsandbytes`):

```python
model = AutoInference.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    backend="transformers",
    load_in_4bit=True,      # BitsAndBytes NF4 quantization
    max_new_tokens=256,
)
```

**8-bit quantization**:

```python
model = AutoInference.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    backend="transformers",
    load_in_8bit=True,
)
```

### ⚡ vLLM

```python
model = AutoInference.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    backend="vllm",
    max_new_tokens=512,
    temperature=0.8,
    tensor_parallel_size=1,        # Number of GPUs
    gpu_memory_utilization=0.9,
)

response = model("Explain attention mechanisms.")
```

### 🦥 Unsloth

```python
model = AutoInference.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    backend="unsloth",
    max_new_tokens=200,
    temperature=0.7,
    max_seq_length=2048,
    load_in_4bit=True,
)

response = model("Write a haiku about autumn.")
```

### 🔧 llama.cpp (GGUF)

```python
# Local GGUF file
model = AutoInference.from_pretrained(
    model_name="/path/to/model.gguf",
    backend="llamacpp",
    max_new_tokens=200,
    temperature=0.6,
    n_gpu_layers=-1,   # -1 = all layers on GPU; 0 = CPU only
    n_ctx=4096,
)

# Download from HuggingFace Hub automatically
model = AutoInference.from_pretrained(
    model_name="",                              # ignored when repo_id is given
    backend="llamacpp",
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
)
```

---

## CLI

Auto-Inference ships with a command-line tool:

```bash
# Single generation
auto-inference generate meta-llama/Meta-Llama-3-8B-Instruct "Hello!"

# Streaming output
auto-inference generate meta-llama/Meta-Llama-3-8B-Instruct "Tell me a story" --stream

# 4-bit quantized
auto-inference generate meta-llama/Meta-Llama-3-8B-Instruct "Hi" --load-in-4bit

# Interactive chat session
auto-inference chat meta-llama/Meta-Llama-3-8B-Instruct --system "You are helpful."

# Streaming chat
auto-inference chat meta-llama/Meta-Llama-3-8B-Instruct --stream

# System info
auto-inference info

# Search HuggingFace Hub
auto-inference search "llama 3 instruct" --limit 5

# Estimate VRAM
auto-inference estimate meta-llama/Meta-Llama-3-8B --dtype float16
```

---

## Utilities

```python
from auto_inference.utils import get_device_info, get_memory_usage, print_system_info

# Print system summary
print_system_info()

# Programmatic device info
info = get_device_info()
print(info["recommended_device"])   # 'auto', 'mps', or 'cpu'
print(info["cuda_devices"])         # list of GPU dicts

# Memory usage (requires psutil for RAM stats)
mem = get_memory_usage()
print(f"GPU: {mem['gpu_allocated_gib']:.2f} GiB")

# Search HuggingFace Hub
from auto_inference.utils import list_hf_models, estimate_model_memory

for m in list_hf_models("mistral 7b", limit=5):
    print(m["model_id"], m["downloads"])

gib = estimate_model_memory("meta-llama/Meta-Llama-3-8B", dtype="float16")
print(f"~{gib:.1f} GiB required")
```

---

## GenerationConfig Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_new_tokens` | int | 512 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0–2) |
| `do_sample` | bool | True | Sampling vs greedy decoding |
| `top_k` | int | 50 | Top-k sampling |
| `top_p` | float | 0.95 | Nucleus (top-p) sampling |
| `repetition_penalty` | float | 1.0 | Penalise token repetition (≥ 1.0) |
| `skip_special_tokens` | bool | True | Strip special tokens from output |
| `stop_sequences` | list[str] | [] | Strings that terminate generation |

---

## Supported Libraries

| Backend | Status | GPU Required |
|---|---|---|
| 🤗 Transformers | ✅ | Optional |
| 🦥 Unsloth | ✅ | Required |
| ⚡ vLLM | ✅ | Required |
| 🔧 llama.cpp | ✅ | Optional |

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

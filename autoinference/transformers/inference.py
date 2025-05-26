from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch


def llm_inference(model_path:str, prompt:str,max_new_tokens, temperature):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = "auto",
            torch_dtype = torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs,max_new_tokens = max_new_tokens , temperature = temperature)
    print(tokenizer.decode(outputs[0], skip_special_tokens = True))


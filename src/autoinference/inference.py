# Code written by Volkan Simsir, 2025.

from transformers import AutoTokenizer
from quantization.gptqmodel import GPTQQuantizer
from quantization.bitsandbytes import BitsAndBytesQuantizer




class AutoInference:
    QUANT_METHODS = {
        'gptq': GPTQQuantizer,
        'bitsandbytes': BitsAndBytesQuantizer,
    }

    def __init__(
        self,
        model_name: str, 
        quant_method: str = 'none',
        quant_config: Dict = None
        ):

        self.model_name = model_name
        self.quantizer = self.QUANT_METHODS[quant_method]()
        self.model = self.quantizer.load_model(model_name, quant_config or {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate(self, prompt: str,**kwargs):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

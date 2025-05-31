from base_quantize import BaseQuantizationMethod

class BitsAndBytesQuantizer(BaseQuantizationMethod):
    def load_model(self, model_name: str, config: Dict[str, Any]):
        try:
           from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        except ImportError:
            raise ImportError("To use GPTQ, install the 'gptqmodel' library: pip install gptqmodel")

        bnb_config = BitsAndBytesConfig(**config)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        return model
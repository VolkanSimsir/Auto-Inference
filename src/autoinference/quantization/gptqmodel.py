from base_quantize import BaseQuantizationMethod
from typing import Any, Dict

class GPTQQuantizer(BaseQuantizationMethod):
    def load_model(self, model_name: str, config: Dict[str, Any]):
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
        except ImportError:
            raise ImportError("To use GPTQ, install the 'gptqmodel' library: pip install gptqmodel")

        bits = config.get("bits", 4)
        group_size = config.get("group_size", 128)
        quant_config = QuantizeConfig(bits=bits, group_size=group_size)
        return GPTQModel.load(model_name, quant_config)
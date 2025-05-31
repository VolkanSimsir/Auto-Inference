from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseQuantizationMethod(ABC): #Soyut sınıf
    @abstractmethod
    def load_model(self, model_name: str, config: Dict[str, Any]):
        pass
    
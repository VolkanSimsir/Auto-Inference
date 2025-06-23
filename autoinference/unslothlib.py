from autoinference.inference import ModelConfig
#from transformers import TextStreamer


class UnslothInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        from unsloth import FastLanguageModel
        self.FastLanguageModel = FastLanguageModel

    def load_model(self):
        """Loads the Unsloth model."""
        model,tokenizer = self.FastLanguageModel.from_pretrained(
            model_name = self.config.model_name,
            max_seq_length = self.config.max_seq_length,
            dtype = None,
            load_in_4bit = self.config.load_in_4bit,
        )
        print("Unsloth model successfully loaded")
        return model, tokenizer
    
    def generate(self, prompt: str):
        model, tokenizer = self.load_model()
        self.FastLanguageModel.for_inference(model) 
        #text_streamer = TextStreamer(tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        _ = model.generate(**inputs, max_new_tokens = self.config.max_new_tokens) #streamer = text_streamer

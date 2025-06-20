from autoinference import AutoInf


model = AutoInf(
        model_name="VolkanSimsir/LLaMA-3-8B-GRPO-math-tr", 
        model_type="transformers")

prompt = "How are you?"
output = model(prompt)
print(output)


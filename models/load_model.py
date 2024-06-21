from models.load_opensource_model import load_opensource_model
from models.load_gemini import load_gemini
def load_model(model_name, quantization, gemini_api_key=""):
    model = None
    tokenizer = None
    config = None
    if model_name in ["GPT-3.5", "GPT-4"]:
        pass
    elif model_name == "claude-3" or model_name == "claude-2.1":
        pass
    elif "gemini" in model_name:
        if not gemini_api_key:
            raise AssertionError("You must specify your gemini API key")
        model = load_gemini(gemini_api_key=gemini_api_key, model_name=model_name)
    else:
        model, tokenizer, config = load_opensource_model(model_name, quantization)
    
    return model, tokenizer, config
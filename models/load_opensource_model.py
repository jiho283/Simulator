import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig

def get_model_dir(model_name:str):
    model_dir = ""
    if model_name == "llama2-7b-chat":
        model_dir = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "llama2-70b-chat":
        model_dir = "meta-llama/Llama-2-70b-chat-hf"
    elif model_name == "tulu2-7b-dpo":
        model_dir = "allenai/tulu-2-dpo-7b"
    elif model_name == "tulu2-70b-dpo":
        model_dir = "allenai/tulu-2-dpo-70b"
    elif model_name == "gemma-2b-it":
        model_dir = "google/gemma-2b-it"
    elif model_name == "gemma-7b-it":
        model_dir = "google/gemma-7b-it"
    elif model_name == "mistral-7b-it":
        model_dir = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_name == "mixtral-it":
        model_dir = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    else:
        return AssertionError("Incorrect model name.")
    return model_dir

def load_opensource_model(model_name:str, quantization:str="no"):
    model_dir = get_model_dir(model_name)
    if quantization == "no":
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True)
    elif quantization == "4bit":
        bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config_4bit, trust_remote_code=True)
    elif quantization == "8bit":
        bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config_8bit, trust_remote_code=True)
    elif quantization == "16bit":
        bnb_config_16bit = BitsAndBytesConfig(load_in_16bit=True, bnb_16bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config_16bit, trust_remote_code=True)
    else:
        raise AssertionError("quantization should be in ['no', '4bit', '8bit', '16bit']")
    if 'tulu' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False, trust_remote_code=True) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True) 
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer, config

def load_opensource_tokenizer(model_name:str):
    model_dir = get_model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True) 
    return tokenizer

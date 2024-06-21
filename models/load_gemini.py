import google.generativeai as genai

def load_gemini(
        candidate_count:int = 1,
        max_output_tokens:int = 256,
        temperature:float = 0.2,
        top_p:float = 0.1,
        gemini_api_key:str = "",
        model_name:str="gemini-pro"
):
    if not gemini_api_key:
        raise AssertionError("You must specify your gemini API key")
    genai.configure(api_key=gemini_api_key)
    generation_config = {
        "candidate_count": candidate_count,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        }
    safety_settings=[
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    if model_name == "gemini-1.5-pro":
        model = genai.GenerativeModel('gemini-1.5-pro', generation_config=generation_config, safety_settings=safety_settings)
    elif model_name == "gemini-pro":
        model = genai.GenerativeModel('gemini-pro', generation_config=generation_config, safety_settings=safety_settings)
    else:
        model = genai.GenerativeModel(model_name, generation_config=generation_config, safety_settings=safety_settings)
    return model
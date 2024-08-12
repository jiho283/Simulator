def gpt_inference(message, model_name, client, temperature=0.2, top_p=0.1):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
        ],
        temperature=temperature,
        top_p = top_p
    )

    return completion.choices[0].message.content


def gpt35_inference(message, client, temperature=0.2, top_p=0.1):
    # deprecated. Use `gpt_inference` instead.
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
        ],
        temperature=temperature,
        top_p = top_p
    )

    return completion.choices[0].message.content

def gpt4_inference(message, client):
    # deprecated. Use `gpt_inference` instead.
    completion = client.chat.completions.create(
                   model="gpt-4o",
                   messages=[
                       {"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": message}
                   ],
                   temperature=0.2,
                   top_p = 0.1
                 )

    return completion.choices[0].message.content

def claude_inference(message, model_name, client):
    model_dir = ""
    if model_name == "claude-3":
        model_dir = "claude-3-opus-20240229"
    elif model_name == "claude-2.1":
        model_dir = "claude-2.1"
    completion = client.messages.create(
                    model=model_dir,
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": message}
                    ],
                    temperature=0.2,
                    top_p=0.1
                )
    return completion.content[0].text

def gemini_inference(message, model):
    response = model.generate_content(message)
    """
    Somehow `response.text` doesn't work (As of April 2024). Using the alternative.
    
    The docs of Gemini says in
    `generativeai/types/generation_types.py`
    that `response.text` is a quick accessor of 
    `self.candidates[0].parts[0].text`.
    Which I believe is a typo of 
    `self.candidates[0].content.parts[0].text`

    """
    result = ""
    try:
        result = response.text
    except:
        try:
            result = response.candidates[0].content.parts[0].text
        except:
            result = "" # Gemini can't generate some responses due to safety reasons. We will consider it as a wrong answer
    return result
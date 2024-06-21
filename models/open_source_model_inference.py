def open_source_model_inference(message, model_name, model, tokenizer, config, output_max_length=128, temperautre=0.2, top_p=0.1):
    """
    Inference for open-source models.
    Tantamount to `gpt35_inference`.

    Parameters:
        message (str): prompt for the model
        model_name (str) : name of the model
        model: model
        tokenizer: tokenizer
        config: configuration
    
    Returns:
        generated_text: generated text of the model. We disregard the input prompt from the output of the model.
    """
    input_ids = None
    # TODO: generalize this.
    if "mistral" in model_name or "mixtral" in model_name or "gemma" in model_name:
        messages = [
        {"role": "user", "content": message},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to("cuda")
        model_inputs_len = model_inputs.shape[1]

        generated_ids = model.generate(model_inputs, max_new_tokens=output_max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)[:, model_inputs_len:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = decoded[0]
        return decoded
    
    elif "tulu" in model_name:
        message = f"<|user|>\n{message}\n<|assistant|>" # <|assistant|>\n
        input_ids = tokenizer(message, return_tensors="pt", max_length=config.max_position_embeddings, truncation=True).to("cuda").input_ids

    elif "llama2" in model_name:
        message = f"[INST] {message} [/INST]"
        input_ids = tokenizer(message, return_tensors="pt", max_length=config.max_position_embeddings, truncation=True).to("cuda").input_ids

    
    input_len = len(input_ids[0])
    generated_ids = None
    try:
        generated_ids = model.generate(input_ids, max_length=input_len+output_max_length, temperature=temperautre, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    except:
        generated_ids = model.generate(input_ids, max_length=input_len+output_max_length, temperature=temperautre, top_p=top_p)
    
    generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

    return generated_text
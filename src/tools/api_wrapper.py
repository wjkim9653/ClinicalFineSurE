import os
import openai
import ast, time, re, json, logging


def inference_api_resolver(inference_spec: dict):
    """
    Input:
        dictionary w/ following keys: 'provider', 'checkpoint', 'endpoint'
            e.g. {"provider":"openai", "checkpoint":"openai/gpt-4.1-mini-2025-04-14", "endpoint":"https://api.openai.com/v1/chat/completions"}
    """
    if any(key not in inference_spec.keys() for key in ["provider", "checkpoint", "endpoint"]):
        logging.ERROR(f"llm inference api resolver failed due to missing key from input parameter dictionary: {key}")

    if inference_spec["provider"].lower() == "openai":
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            logging.error("Error while trying to fetch OS ENV VAR for API KEY: OPENAI_API_KEY\n{e}")
        
        client = openai.OpenAI(api_key=openai_api_key)
        model_ckpt = inference_spec["checkpoint"]
        return client, model_ckpt
    
    if inference_spec["provider"].lower() == "openrouter":
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        except:
            logging.error("Error while trying to fetch OS ENV VAR for API KEY: OPENROUTER_API_KEY\n{e}")
        base_url = inference_spec["endpoint"]
        client = openai.OpenAI(
            base_url=base_url,
            api_key=openrouter_api_key
        )
        model_ckpt = inference_spec["checkpoint"]
        return client, model_ckpt





def get_openai_response(client, prompt, model, temperature=0.0):

    ''' A function to get the response from GPT-series
    Args:
        client: openai client
        prompt: input prompt
        model: openai model name
    Return: 
        text_response: the output from LLMs
    '''
    if "/" in model:
        provider, _, model_name = model.partition("/")
        if provider == "openai":
            model = model_name
        else:
            model = model
    elif "--" in model:
        provider, _, model_name = model.partition("--")
        if provider == "openai":
            model = model_name
        else:
            model = f"{provider}/{model_name}"
    
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    if all(reasoning_model_name not in model for reasoning_model_name in ['o1', 'o3', 'o4']):
        params["temperature"] = temperature
    
    response = client.chat.completions.create(**params)
    text_response = response.choices[0].message.content

    return text_response
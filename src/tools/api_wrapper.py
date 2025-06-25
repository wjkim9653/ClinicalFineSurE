import openai
import ast, time, re, json, logging

def get_openai_response(client, prompt, model, temperature=0.0):

    ''' A function to get the response from GPT-series
    Args:
        client: openai client
        prompt: input prompt
        model: openai model name
    Return: 
        text_response: the output from LLMs
    '''

    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    if all(reasoning_model_name not in model for reasoning_model_name in ['o1', 'o3', 'o4']):
        params["temperature"] = temperature
    
    response = client.chat.completions.create(**params)
    text_response = response.choices[0].message.content

    return text_response
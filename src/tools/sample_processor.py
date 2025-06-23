import time, re, json, logging
from ClinicalFineSurE.src.tools.api_wrapper import get_openai_response

def keyfact_extraction(client, prompt, model, temperature=0.0, max_retries=2):
    '''
    returns:
        llm_output: str
        keyfacts: list (of str)
    '''
    def parse_llm_output_to_list(llm_output_str):
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", llm_output_str.strip(), flags=re.DOTALL)
        try:
            parsed_json = json.loads(cleaned)
            return parsed_json.get("key facts", [])
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error: {e}")
            return []

    for attempt in range(1, max_retries+1):
        try:
            llm_output = get_openai_response(client, prompt, model, temperature)
            keyfacts = parse_llm_output_to_list(llm_output_str=llm_output)
            if keyfacts:
                return llm_output, keyfacts
            else:
                raise ValueError("Parsed KeyFacts List is Empty")
        except Exception as e:  # 실패할 경우 재시도
            logging.warning(f"[Attempt {attempt}] KeyFact-Extraction Fialed: {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                logging.error("All Retry Attempts Failed")
                return "", []  # fallback
    return "", []  # fallback




class SampleProcessor:
    def __init__(self):
        pass
import os
from pathlib import Path
import csv
import time, re, json, logging
from ClinicalFineSurE.src.tools.api_wrapper import get_openai_response
from ClinicalFineSurE.src.tools.utils import simplify_checkpoint

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

def read_csv(filepath: str | Path):
    """
    Read; Row-by-Row; the CSV file
    """
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # csv.DictReader -> Read CSV rows as dict (can access w/ row name instead of idx)
        for row in reader:
            yield row  # yield iterable

def create_transcript_files(
        tag: str, 
        original_file: str | Path, 
        out_path: str | Path
        ):
    original_file = Path(original_file)
    out_path = Path(out_path)
    file_out = out_path / f"{tag}_transcripts.jsonl"  # single file as output

    with open(file_out, 'w', encoding='utf-8') as f_out:
        for row in read_csv(filepath=original_file):
            sample_id = f"{tag}_{row['ID']}"
            sample_transcript = row["dialogue"]
            new_row = {
                "sample_id": sample_id,
                "transcript": sample_transcript
            }
            f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')

    return file_out

def generate_keyfact_list_files(
        tag: str, 
        transcript_file: str | Path, 
        out_path: str | Path,
        pseudo_labeler_specs: list[dict]  # config["pseudo-labeler"]["spec"] as parameter
        ):
    transcript_file = Path(transcript_file)
    out_path = Path(out_path)

    out_paths = []
    
    for pseudo_labeler_spec_dict in pseudo_labeler_specs:  # iterate over each dict from `specs` list
        simple_identifier_for_llm = simplify_checkpoint(pseudo_labeler_spec_dict["checkpoint"])
        file_out = out_path / f"{tag}_keyfact_by_{simple_identifier_for_llm}.jsonl"
        """구현해라 원진아"""
        """여기서부터 호출 관련 로직 넣고...."""

        out_paths.append(file_out)
    
    return out_paths

def generate_summary_files(
        tag: str, 
        transcript_file_path: str | Path, 
        out_path: str | Path,
        summarizer_lm_specs: list[dict]  # config["summarizer"]["spec"] as parameter
    ):
    transcript_file_path = Path(transcript_file_path)
    out_path = Path(out_path)  # ⚠️ directory, not a file
    
    out_paths = []

    for summarizer_spec_dict in summarizer_lm_specs:  # iterate over each summarizer
        simple_identifier_for_summarizer = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
        file_out = out_path / f"{tag}_summary_by_{simple_identifier_for_summarizer}.jsonl"
        """구현해라 원진아"""
        """여기서부터 호출 관련 로직 넣고...."""

        out_paths.append(file_out)
    
    return out_paths

def generate_factuality_files(
        tag: str,
        transcript_file_path: str | Path,
        summary_file_paths: list[str | Path],
        out_path: str | Path
    ):
    """
    on each summary sentences list (for a corresponding transcript sample),
    generate (pseudo-)factuality-labels on each summary sentences as binary(1,0), by checking whether it's grounded on the corresponding transcript.
    also, generate (psudo-)factuality_types on each summary sentences from given types
        `ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']`
    """




class SampleProcessor:
    def __init__(self):
        pass
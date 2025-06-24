import os
from pathlib import Path
import time, re, json, logging
from ClinicalFineSurE.src.tools.api_wrapper import get_openai_response
from ClinicalFineSurE.src.tools.utils import simplify_checkpoint, read_csv, load_jsonl

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

def create_transcript_files(
        tag: str, 
        original_file: str | Path, 
        out_path: str | Path
    ):
    original_file = Path(original_file)
    out_path = Path(out_path)
    file_out = out_path / f"{tag}_transcript.jsonl"  # single file as output

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
    if not os.path.exists(transcript_file):  # if transcript file not found
        logging.error(f"Failed to find transcript jsonl file at: {transcript_file}")

    out_path = Path(out_path)
    out_paths = []
    
    for pseudo_labeler_spec in pseudo_labeler_specs:  # iterate over each dict from `specs` list (⚠️ keep at 1 for now)
        labeler_identifier = simplify_checkpoint(pseudo_labeler_spec["checkpoint"])
        file_out = out_path / f"{tag}_keyfact_by_{labeler_identifier}.jsonl"
        """구현해라 원진아"""
        """여기서부터 호출 관련 로직 넣고...."""

        out_paths.append(file_out)
    assert len(out_paths) == len(pseudo_labeler_specs)

def generate_summary_files(
        tag: str, 
        transcript_file: str | Path, 
        out_path: str | Path,
        summarizer_lm_specs: list[dict]  # config["summarizer"]["spec"] as parameter
    ):
    transcript_file = Path(transcript_file)
    if not os.path.exists(transcript_file):  # if transcript file not found
        logging.error(f"Failed to find transcript jsonl file at: {transcript_file}")
    
    out_path = Path(out_path)  # ⚠️ directory, not a file
    out_paths = []

    for summarizer_spec_dict in summarizer_lm_specs:  # iterate over each summarizer
        summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
        file_out = out_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
        """구현해라 원진아"""
        """여기서부터 호출 관련 로직 넣고...."""

        out_paths.append(file_out)
    assert len(out_paths) == len(summarizer_lm_specs)

def generate_factuality_files(
        tag: str,
        transcript_file: str | Path, 
        summary_file_path: str | Path,  # parent dir for where summary jsonl files are located at
        out_path: str | Path,
        pseudo_labeler_specs: list[dict],  # config["pseudo-labeler"]["spec"] as parameter
        summarizer_lm_specs: list[dict]  # config["summarizer"]["spec"] as parameter
    ):
    """
    on each summary sentences list (for a corresponding transcript sample),
    generate (pseudo-)factuality-labels on each summary sentences as binary(1,0), by checking whether it's grounded on the corresponding transcript.
    also, generate (psudo-)factuality_types on each summary sentences from given types
        `ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']`
    """
    transcript_file = Path(transcript_file)
    summary_file_path = Path(summary_file_path)
    out_path = Path(out_path)

    if not os.path.exists(transcript_file):  # if transcript file not found
        logging.error(f"Failed to find transcript jsonl file at: {transcript_file}")

    for pseudo_labeler_spec in pseudo_labeler_specs:  # for each Pseudo-Labeler LLMs (⚠️ keep at 1 for now)
        # use the LLM spec to conduct pseudo-labeling
        # ⚠️ Set up the client for corresponding LLM
        for summarizer_spec_dict in summarizer_lm_specs:  # for each Summary from differing Summarization LMs
            summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
            summary_file_path = summary_file_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
            if not os.path.exists(summary_file_path): # corresponding summary jsonl file doesn't exist
                logging.error(f"Failed to find summary jsonl file for {summarizer_identifier} at: {summary_file_path}")
            """
            구현해라 원진아
            각 summary 까서 샘플id로 상응하는 transcript 가져오고 llm으로 factuality label이랑 factuality type 생성하렴
            """

def generate_alignment_files(
        tag: str, 
        keyfact_file_path: str | Path, 
        summary_file_path: str | Path, 
        out_path: str | Path, 
        pseudo_labeler_specs: list[dict],  # config["pseudo-labeler"]["spec"] as parameter
        summarizer_lm_specs: list[dict]  # config["summarizer"]["spec"] as parameter
    ):
    """
    on each summary sentences, compare against corresponding keyfact list.
    using pseudo-labeler llm, generate 
    a. keyfact_labels(binary int list, each elem corresponding to a keyfact from keyfacts list)
    & 
    b. sentence_labels(binary int list, each elem corresponding to a sentence from summaries list)
    """
    keyfact_file_path = Path(keyfact_file_path)
    summary_file_path = Path(summary_file_path)
    out_path = Path(out_path)
    
    for pseudo_labeler_spec in pseudo_labeler_specs:  # for each Pseudo-Labeler LLMs (⚠️ keep at 1 for now)
        labeler_identifier = simplify_checkpoint(pseudo_labeler_spec["checkpoint"])
        keyfact_file = out_path / f"{tag}_keyfact_by_{labeler_identifier}.jsonl"
        if not os.path.exists(keyfact_file): # corresponding keyfact list jsonl file doesn't exist
            logging.error(f"Failed to find keyfact list jsonl file for {labeler_identifier} at: {keyfact_file}")
        keyfact_list = load_jsonl(keyfact_file)  # list of dict -> ⚠️ use `sample_id` to match with summary_list
        
        # use the LLM spec to conduct pseudo-labeling
        # ⚠️ Set up the client for corresponding LLM

        for summarizer_spec_dict in summarizer_lm_specs:  # for each Summary from differing Summarization LMs
            summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
            summary_file = summary_file_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
            if not os.path.exists(summary_file): # corresponding summary jsonl file doesn't exist
                logging.error(f"Failed to find summary jsonl file for {summarizer_identifier} at: {summary_file}")
            summary_list = load_jsonl(summary_file)  # list of dict -> ⚠️ use `sample_id` to match with keyfact_list
            
            """
            구현해라 원진아
            각 summary 까서 샘플id로 상응하는 transcript 가져오고 llm으로 factuality label이랑 factuality type 생성하렴
            """
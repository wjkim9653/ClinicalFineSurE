import os
import pprint
from pathlib import Path
import time, re, json, logging
from src.tools.api_wrapper import *
from src.tools.utils import *
from src.tools.lm_prompt_builder import *
from src.tools.lm_response_parser import *


def keyfact_extraction(client, prompt, model, temperature=0.0, max_retries=2):
    '''
    returns:
        llm_output: str
        keyfacts: list (of str)
    '''
    def parse_llm_output_to_list(llm_output_str):
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", llm_output_str.strip(), flags=re.DOTALL)  # .DOTALL is for including linebreaks('\n') in the resulting regex extraction strings
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
    """
    STEP 1: create Transcript files
    This part is simple formatting & re-structuring process
    When complete, this part will yield a transcript file in json format({dataset_name}_transcript.json), containing 'sample_id'(str) and 'transcript'(str) as keys.
    """
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
    """
    STEP 2: generate Key-Fact Lists for each sample in Transcript
    This part is a Pseudo-Labeling process, and should be done w/ SOTA LLM
    When complete, this part will yield a Pseudo-Labeled Key-Fact List file in json format({llm_name}_keyfact.json), containing 'sample_id'(str), 'keyfact'(str) and 'keyfact_list'(list of str) as keys.
    """
    transcript_file = Path(transcript_file)
    if not os.path.exists(transcript_file):  # if transcript file not found
        logging.error(f"Failed to find transcript jsonl file at: {transcript_file}")

    out_path = Path(out_path)
    out_paths = []
    
    for pseudo_labeler_spec in pseudo_labeler_specs:  # iterate over each dict from `specs` list (⚠️ keep at 1 for now)
        client, model_ckpt = inference_api_resolver(pseudo_labeler_spec)
        
        labeler_identifier = simplify_checkpoint(model_ckpt)
        file_out = out_path / f"{tag}_keyfact_by_{labeler_identifier}.jsonl"
        
        with open(file_out, 'w', encoding='utf-8') as f_out:
            for row in read_jsonl(filepath=transcript_file):
                print(row)
                sample_id = row["sample_id"]
                transcript = row["transcript"]
                
                keyfact_extraction_prompt = get_keyfact_extraction_prompt(transcript)
                try:
                    llm_output, keyfacts = keyfact_extraction(
                        client=client,
                        prompt=keyfact_extraction_prompt,
                        model=model_ckpt,
                    )  # Acquire Machine-Extracted-KeyFacts via LLM API Call

                    new_row = {
                        "sample_id": sample_id,
                        "transcript": transcript,  # ⚠️ rm if unnecessary
                        "keyfact": llm_output,  # ⚠️ rm if unnecessary
                        "keyfact_list": keyfacts
                    }
                    f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')  # save newly extracted keyfact instance as new row
                    logging.info(f"Successfully saved newly extracted Machine-KeyFact for sample_id: {sample_id}")
                except Exception as e:
                    logging.error(f"KeyFact Extraction for sample_id({sample_id}) Failed :\n{e}")

        out_paths.append(file_out)
    assert len(out_paths) == len(pseudo_labeler_specs)


def generate_summary_files(
        tag: str, 
        transcript_file: str | Path, 
        out_path: str | Path,
        summarizer_lm_specs: list[dict]  # config["summarizer"]["spec"] as parameter
    ):
    """
    STEP 3: generate Summaries for each sample in Transcript
    This part is a Sample Generation process, and must be done w/ various Summarization Models or LMs
    When complete, this part will yield a number of summary files in json formats({summarizer_lm_name}_summary.json), each containing 'sample_id'(str), 'summarizer'(str), 'summary'(str), 'summary_list'(list of str) as keys.
    """
    transcript_file = Path(transcript_file)
    if not os.path.exists(transcript_file):  # if transcript file not found
        logging.error(f"Failed to find transcript jsonl file at: {transcript_file}")
    
    out_path = Path(out_path)  # ⚠️ directory, not a file
    out_paths = []

    for summarizer_spec_dict in summarizer_lm_specs:  # iterate over each summarizer
        client, model_ckpt = inference_api_resolver(inference_spec=summarizer_spec_dict)
        summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
        file_out = out_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
        with open(file_out, 'w', encoding='utf-8') as f_out:
            for row in read_jsonl(filepath=transcript_file):
                sample_id = row["sample_id"]
                transcript = row["transcript"]
                summarization_prompt = get_summarization_prompt(transcript=transcript)
                try:
                    raw_summarization_output = get_openai_response(client=client, prompt=summarization_prompt, model=model_ckpt)
                    parsed_summary_string, parsed_summary_list = parsing_summarizer_output(output=raw_summarization_output)
                    new_row = {
                        "sample_id": sample_id,
                        "transcript": transcript,
                        "summarizer": summarizer_identifier,
                        "summary": parsed_summary_string,
                        "summary_list": parsed_summary_list
                    }
                    f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')
                    logging.info(f"Successfully saved newly generated summary for sample_id({sample_id}) w/ model({summarizer_identifier})")
                except Exception as e:
                    logging.error(f"Error while generating summary for sample_id({sample_id}) w/ model({summarizer_identifier}):\n{e}")
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
        client, model_ckpt = inference_api_resolver(pseudo_labeler_spec)  # use the LLM spec to conduct pseudo-labeling
        labeler_identifier = simplify_checkpoint(model_ckpt)
        out_paths = []
        for summarizer_spec_dict in summarizer_lm_specs:  # for each Summary from differing Summarization LMs
            summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
            unique_summary_file_path = summary_file_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
            if not os.path.exists(unique_summary_file_path): # corresponding summary jsonl file doesn't exist
                logging.error(f"Failed to find summary jsonl file for {summarizer_identifier} at: {unique_summary_file_path}")
            """
            각 summary 까서 샘플id로 상응하는 transcript 가져오고 llm으로 factuality label이랑 factuality type 생성
            """
            file_out = out_path / f"{tag}_{summarizer_identifier}_summary_factuality_by_{labeler_identifier}.json"
            with open(file_out, 'w', encoding='utf-8') as f_out:
                for row in read_jsonl(filepath=unique_summary_file_path):  # iter over each transcript-summary pairs
                    retry_cnt = 0
                    while (retry_cnt < 3):
                        try:
                            sample_id = row["sample_id"]
                            transcript = row["transcript"]
                            summary_list = row["summary_list"]

                            fact_checking_prompt = get_fact_checking_prompt(input=transcript, sentences=summary_list)
                            factuality_raw_llm_output = get_openai_response(client=client, prompt=fact_checking_prompt, model=model_ckpt)
                            factuality_labels, factuality_types = parsing_llm_fact_checking_output(factuality_raw_llm_output)
                            
                            # Checking if parsing is successful
                            if len(factuality_labels) == len(summary_list):  # Successful parsing
                                new_row = {
                                    "sample_id": sample_id,
                                    "transcript": transcript,
                                    "summary_list": summary_list,
                                    "summarizer": summarizer_identifier,
                                    "labeler": labeler_identifier,
                                    "factuality_labels": factuality_labels,
                                    "factuality_types": factuality_types
                                }
                                f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')  # save newly generated & parsed factuaity labels & types
                                logging.info(f"Successfully saved newly extracted Factuality Labels & Types for sample_id: {sample_id}")
                                break
                            retry_cnt += 1
                        except Exception as e:
                            logging.error(f"Pseudo-Labeling Factuality Labels & Types for sample_id({sample_id}) Failed :\n{e}")
                            retry_cnt += 1
            out_paths.append(out_path)
        assert len(out_paths) == len(summarizer_lm_specs)
        

def generate_alignment_files(
        tag: str,
        transcript_file_path: str | Path,
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
    transcript_file_path = Path(transcript_file_path)
    keyfact_file_path = Path(keyfact_file_path)
    summary_file_path = Path(summary_file_path)
    out_path = Path(out_path)
    
    transcript_file = transcript_file_path / f"{tag}_transcript.jsonl"
    all_sample_ids = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            all_sample_ids.append(sample["sample_id"])

    for pseudo_labeler_spec in pseudo_labeler_specs:  # for each Pseudo-Labeler LLMs (⚠️ keep at 1 for now)
        client, model_ckpt = inference_api_resolver(inference_spec=pseudo_labeler_spec)
        labeler_identifier = simplify_checkpoint(pseudo_labeler_spec["checkpoint"])
        keyfact_file = keyfact_file_path / f"{tag}_keyfact_by_{labeler_identifier}.jsonl"
        if not os.path.exists(keyfact_file): # corresponding keyfact list jsonl file doesn't exist
            logging.error(f"Failed to find keyfact list jsonl file for {labeler_identifier} at: {keyfact_file}")
        keyfacts = load_jsonl(keyfact_file)  # list of dict -> ⚠️ use `sample_id` to match with summary_list
        indexed_keyfacts = {item["sample_id"]: item for item in keyfacts}

        file_out = out_path / f"{tag}_{labeler_identifier}_keyfact_alignment_against_summaries.json"
        with open(file_out, 'w', encoding='utf-8') as f_out:
            for summarizer_spec_dict in summarizer_lm_specs:  # for each Summary from differing Summarization LMs
                summarizer_identifier = simplify_checkpoint(summarizer_spec_dict["checkpoint"])
                summary_file = summary_file_path / f"{tag}_summary_by_{summarizer_identifier}.jsonl"
                if not os.path.exists(summary_file): # corresponding summary jsonl file doesn't exist
                    logging.error(f"Failed to find summary jsonl file for {summarizer_identifier} at: {summary_file}")
                summaries = load_jsonl(summary_file)  # list of dict -> ⚠️ use `sample_id` to match with keyfact_list
                indexed_summaries = {item["sample_id"]: item for item in summaries}

                for sample_id in all_sample_ids:
                    keyfact_list = indexed_keyfacts.get(sample_id)["keyfact_list"]
                    summary_list = indexed_summaries.get(sample_id)["summary_list"]
                    try:
                        keyfact_alignment_prompt = get_keyfact_alighment_prompt(keyfacts=keyfact_list, sentences=summary_list)
                        raw_output = get_openai_response(client=client, prompt=keyfact_alignment_prompt, model=model_ckpt)
                        keyfact_alignment_labels, aligned_summary_line_numbers = parsing_llm_keyfact_alighment_output(output=raw_output)
                        '''Score Calculations' omitted here, since it's about creating pseudo-labeled dataset for FineSurE input, not the final score and rankings.'''
                        new_row = {
                            "sample_id": sample_id,
                            "keyfact_labeler": labeler_identifier,
                            "summarizer": summarizer_identifier,
                            "labler": labeler_identifier,
                            "keyfact_list": keyfact_list,
                            "summary_list": summary_list,
                            "keyfact_alignment_labels": keyfact_alignment_labels,
                            "matched_summary_lines": aligned_summary_line_numbers
                        }
                        f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')
                        logging.info(f"Successfully saved newly labled Keyfact <-> Summary Alignment for sample_id: {sample_id}")
                    except Exception as e:
                        logging.error(f"Error while pseudo-labeling keyfact-summary alignment for sample_id({sample_id}):\n{e}")

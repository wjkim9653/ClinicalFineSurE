import yaml
import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from ClinicalFineSurE.src.tools.api_wrapper import *
from ClinicalFineSurE.src.tools.lm_prompt_builder import *
from ClinicalFineSurE.src.tools.sample_processor import SampleProcessor

def setup_logger(level: str):
    """
    Set-Up; Logger Configurations
    """
    logging.basicConfig(
        level=logging.ERROR if level == "error" else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("preprocess_mts_dialog.log"),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """
    Parse; Command-Line Arguments
    """
    parser = argparse.ArgumentParser(description="Preprocess CSV Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file for this run"
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="info",
        help="set the logging level; options: 'info', 'error'"
    )
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def read_csv(filepath: str):
    """
    Read; Row-by-Row; the CSV file
    """
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # csv.DictReader -> Read CSV rows as dict (can access w/ row name instead of idx)
        for row in reader:
            yield row  # yield iterable

def process_row(row: Dict[str, Any], processor: SampleProcessor) -> Dict[str, Any]:
    """
    Process; a single Row; using an External Processor class instance
    """
    try:
        result = processor.process(row)
        return result
    except Exception as e:
        logging.error(f"Error processing row w/ id={row.get("ID")}")  # ⚠️ original csv's column-name dependency here! ideally, such hard-coded dependency("ID") must be removed. i leave it to future work...
        fallback_result = processor.fallback_process(row)  # some kind of fallback when there's an error. i guess return some dict that share the same structure(k-v) but w/ placeholder values? idk will implement later. think i should include 'error' field of something.
        return fallback_result
    
def create_transcript_files(tag: str, file_in: str | Path, file_out: str | Path):
    file_in = Path(file_in)
    file_out = Path(file_out)

    with open(file_out, 'w', encoding='utf-8') as f_out:
        for row in read_csv(filepath=file_in):
            sample_id = f"{tag}_{row['ID']}"
            sample_transcript = row["dialogue"]
            new_row = {
                "sample_id": sample_id,
                "transcript": sample_transcript
            }
            f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')

    return file_out
     

def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logger(args.logging)

    # Making sure Output Paths exists
    output_paths = config["output_paths"]  # dict
    for output_descriptor, output_path in output_paths.items():
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to check/create output directory for {output_descriptor} at: {output_path}")

    # Processing Input Files
    input_paths = config["input_paths"]  # dict
    for input_descriptor, input_path in input_paths.items():
        if not os.path.exists(input_path):
            logging.error(f"Failed to find input csv file for {input_descriptor} at: {input_path}")
            continue
        logging.info(f"Found input csv file for {input_descriptor} at: {input_path}")

        # STEP 1: create Transcript files
            # This part is simple formatting & re-structuring process
                # When complete, this part will yield a transcript file in json format({dataset_name}_transcript.json), containing 'sample_id'(str) and 'transcript'(str) as keys.
        

        # STEP 2: generate Key-Fact Lists for each sample in Transcript
            # This part is a Pseudo-Labeling process, and must be done w/ SOTA LLM
            # When complete, this part will yield a Pseudo-Labeled Key-Fact List file in json format({llm_name}_keyfact.json), containing 'sample_id'(str), 'keyfact'(str) and 'keyfact_list'(list of str) as keys.


        # STEP 3: generate Summaries for each sample in Transcript
            # This part is a Sample Generation process, and must be done w/ various Summarization Models or LMs
            # When complete, this part will yield a number of summary files in json formats({summarizer_lm_name}_summary.json), each containing 'sample_id'(str), 'summarizer'(str), 'summary'(str), 'summary_list'(list of str) as keys.


        # STEP 4: generate Factuality Labels & Factuality Types for Each Transcript + Summary samples from Transcript & Summary
            # This part is a Pseudo-Labeling process, and must be done w/ SOTA LLM
            # When complete, this part will yield a number of Factuality Label and Types (between Transcript and Summary Sentences) for each summary files (from each summarizers), each containing 'sample_id'(str), 'summarizer'(str), 'factuality_labels'(list of int), 'factuality_types'(list of str) as keys.


        # STEP 5: generate Alignments between Summaries and corresponding Key-Fact Lists
            # This part is a Pseudo-Labeling process, and must be done w/ SOTA LLM
            # When complete, this part will yield a number of Alignment Files (between KeyFact List and Summary Sentences) for each summary files (from each summarizers), each containing 'sample_id'(str), 'summarizer'(str), 'keyfact_labels'(list of int), 'sentence_labels'(list of int) as keys.
    
    
    
    
    
    
    # Instanciate processor class ⚠️ This should be much more complex than this. maybe pass yaml w/ configs in it?
    processor = SampleProcessor(config=config)

    processed_results = []
    for row in read_csv(csv_path):
        processed = process_row(row, processor)
        processed_results.append(processed)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Finished processing. Output saved to: {output_path}")



if __name__ == "__main__":
    main()


"""
To Run: 
python preprocess/preprocess-mts-dialog.py dataset/sampled/MTS-Dialog.csv --config configs/config-example.yaml
"""
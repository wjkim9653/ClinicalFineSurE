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
from ClinicalFineSurE.src.tools.sample_processor import (create_transcript_files, generate_summary_files, generate_keyfact_list_files, generate_factuality_files, generate_alignment_files)


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
        transcript_file = create_transcript_files(
            tag=config["tag"],
            original_file=input_path,
            out_path=output_paths["transcript"]
        )

        # STEP 2: generate Key-Fact Lists for each sample in Transcript
            # This part is a Pseudo-Labeling process, and should be done w/ SOTA LLM
            # When complete, this part will yield a Pseudo-Labeled Key-Fact List file in json format({llm_name}_keyfact.json), containing 'sample_id'(str), 'keyfact'(str) and 'keyfact_list'(list of str) as keys.
        generate_keyfact_list_files(
            tag=config["tag"],
            transcript_file=transcript_file,
            out_path=output_paths["keyfact"],
            pseudo_labeler_specs=config["pseudo-labeler"]["spec"]  # list of dict as parameter
        )

        # STEP 3: generate Summaries for each sample in Transcript
            # This part is a Sample Generation process, and must be done w/ various Summarization Models or LMs
            # When complete, this part will yield a number of summary files in json formats({summarizer_lm_name}_summary.json), each containing 'sample_id'(str), 'summarizer'(str), 'summary'(str), 'summary_list'(list of str) as keys.
        generate_summary_files(
            tag=config["tag"],
            transcript_file=transcript_file,
            out_path=output_paths["summary"],
            summarizer_lm_specs=config["summarizer"]["spec"]
        )

        # STEP 4: generate Factuality Labels & Factuality Types for Each Transcript + Summary samples from Transcript & Summary
            # This part is a Pseudo-Labeling process, and must be done w/ SOTA LLM
            # When complete, this part will yield a number of Factuality Label and Types (between Transcript and Summary Sentences) for each summary files (from each summarizers), each containing 'sample_id'(str), 'summarizer'(str), 'factuality_labels'(list of int), 'factuality_types'(list of str) as keys.
        generate_factuality_files(
            tag=config["tag"],
            transcript_file=transcript_file,
            summary_file_path=output_paths["summary"],
            out_path=output_paths["factuality"],
            pseudo_labeler_specs=config["pseudo-labeler"]["spec"],
            summarizer_lm_specs=config["summarizer"]["spec"]
        )

        # STEP 5: generate Alignments between Summaries and corresponding Key-Fact Lists
            # This part is a Pseudo-Labeling process, and must be done w/ SOTA LLM
            # When complete, this part will yield a number of Alignment Files (between KeyFact List and Summary Sentences) for each summary files (from each summarizers), each containing 'sample_id'(str), 'summarizer'(str), 'keyfact_labels'(list of int), 'sentence_labels'(list of int) as keys.
        generate_alignment_files(
            tag=config["tag"], 
            keyfact_file_path=output_paths["keyfact"],
            summary_file_path=output_paths["summary"],
            out_path=config["alignment"],
            pseudo_labeler_specs=config["pseudo-labeler"]["spec"],
            summarizer_lm_specs=config["summarizer"]["spec"]
        )
    
    
    logging.info(f"Finished processing. Output saved to: {output_path}")



if __name__ == "__main__":
    main()


"""
To Run: 
python preprocess/preprocess-mts-dialog.py dataset/sampled/MTS-Dialog.csv --config configs/config-example.yaml
"""
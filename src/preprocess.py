import yaml
import argparse
import csv
import json
import logging
import os
from typing import Any, Dict

from ClinicalFineSurE.src.tools.api_wrapper import *
from ClinicalFineSurE.src.tools.lm_prompt_builder import *
from ClinicalFineSurE.src.tools.sample_processor import SampleProcessor

def setup_logger():
    """
    Set-Up; Logger Configurations
    """
    logging.basicConfig(
        level=logging.INFO,
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
    parser = argparse.ArgumentParser(description="Preprocess MTS-Dialog CSV Dataset.")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the original MTS-Dialog CSV file (e.g., dataset/original/MTS-Dialog-TrainingSet.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to save the output JSON file (after pre-processing)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file for this run"
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
    

def main():
    setup_logger()
    args = parse_args()

    logging.info(f"Starting pre-processing w/ CSV: {args.csv_path}")
    config = load_config(args.config)

    output_dir = args.output_dir or config["output"]["directory"]  # if cmd-line arg given, override the output-dir in config
    csv_path = args.csv_path

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, "processed_data.json")  # ⚠️ don't think this is a clever way to name output files... will come back later

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
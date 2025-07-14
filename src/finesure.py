import os
import yaml
import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict

from src.tools.api_wrapper import *
from src.tools.lm_prompt_builder import *
from src.tools.sample_processor import *
from src.tools.scoring import *

def parse_args():
    """
    Parse; Command-Line Arguments
    """
    parser = argparse.ArgumentParser(description="Run FineSurE Pipeline through Pre-processed Dataset")
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

def data_aggregator(
    tag: str,
    factuality_file_path: str | Path,
    alignment_file_path: str | Path,
    out_path: str | Path,
    pseudo_labeler_specs: list[dict],
    summarizer_lm_specs: list[dict]
):
    factuality_file_path = Path(factuality_file_path)
    alignment_file_path = Path(alignment_file_path)
    out_path = Path(out_path)
    
    for pseudo_labeler_spec in pseudo_labeler_specs:
        pseudo_labeler_identifier = simplify_checkpoint(pseudo_labeler_spec["checkpoint"])  # 'openai--gpt-4.1-nano-2025-04-14'
        for summarizer_lm_spec in summarizer_lm_specs:
            summarizer_lm_identifier = simplify_checkpoint(summarizer_lm_spec["checkpoint"])  # 'qwen--qwen3-8b'

            # Resolve Filenames for (4)_factuality and (5)_alignment
            factuality_file_name = factuality_file_path / f"{tag}_{summarizer_lm_identifier}_summary_factuality_by_{pseudo_labeler_identifier}.json"
            alignment_file_name = alignment_file_path / f"{tag}_{pseudo_labeler_identifier}_keyfact_alignment_against_summaries.json"

            # get all sample_ids for iteration purpose
            all_sample_ids = []
            with open(factuality_file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    all_sample_ids.append(sample["sample_id"])

            # load pre-processed datasets
            factuality_dicts = load_jsonl(filepath=factuality_file_name)
            alignment_dicts = load_jsonl(filepath=alignment_file_name)
            factuality_indexed = {
                f"{d['sample_id']}_{d['summarizer']}": d
                for d in factuality_dicts
            }
            alignment_indexed = {
                f"{d['sample_id']}_{d['summarizer']}": d
                for d in alignment_dicts
            }

            # aggregating & creating new rows of dicts to iter over
            for sample_id in all_sample_ids:
                key = f"{sample_id}_{summarizer_lm_identifier}"
                
                factuality_instance = factuality_indexed[key]
                alignment_instance = alignment_indexed[key]
                
                aggregated_new_row = alignment_instance
                aggregated_new_row["transcript"] = factuality_instance["transcript"]
                aggregated_new_row["gt_labels"]["factuality_labels"] = factuality_instance["factuality_labels"]
                aggregated_new_row["gt_labels"]["factuality_types"] = factuality_instance["factuality_types"]

                yield aggregated_new_row

def finesure_pipeline(
        judge_llm_spec: dict,
        sample: dict
    ):
    """
    Summary List를 보고 Factuality_Labels, Factuality_Types를 예측
    +
    Summary List와 KeyFact List를 동시에 보고 Keyfact_Alignment_Labels, Matched_Summary_Labels, Matched_Summary_Lines를 예측

    Pred Labels들이 추가된 new_row dict를 반환
    """
    # judge llm inference resolve
    client, model_ckpt = inference_api_resolver(inference_spec=judge_llm_spec)

    # factuality prediction 수행
    fact_checking_prompt = get_fact_checking_prompt(input=sample['transcript'], sentences=sample['summary_list'])
    factuality_raw_llm_output = get_openai_response(client=client, prompt=fact_checking_prompt, model=model_ckpt)
    factuality_labels, factuality_types = parsing_llm_fact_checking_output(output=factuality_raw_llm_output)
    assert len(factuality_labels) == len(sample["summary_list"])  # Successful Parsing
    assert len(factuality_types) == len(sample["summary_list"])  # Successful Parsing
    
    # alignment prediction 수행
    keyfact_alighment_prompt = get_keyfact_alighment_prompt(keyfacts=sample["keyfact_list"], sentences=sample["summary_list"])
    alignment_raw_llm_output = get_openai_response(client=client, prompt=keyfact_alighment_prompt, model=model_ckpt)
    keyfact_alignment_labels, matched_summary_lines = parsing_llm_keyfact_alighment_output(output=alignment_raw_llm_output)
    assert len(keyfact_alignment_labels) == len(sample["keyfact_list"])
    matched_summary_labels = [0 for _ in range(len(sample["summary_list"]))]
    # print(matched_summary_labels)
    # print(matched_summary_lines)
    for summary_line_number in matched_summary_lines:
        matched_summary_labels[summary_line_number-1] = 1
    assert len(matched_summary_labels) == len(sample["summary_list"])
    
    pred_labels = {
        "keyfact_alignment_labels": keyfact_alignment_labels,
        "matched_summary_labels": matched_summary_labels,
        "matched_summary_lines": matched_summary_lines,
        "factuality_labels": factuality_labels,
        "factuality_types": factuality_types
    }
    new_row = sample
    new_row["pred_labels"] = pred_labels
    return new_row

def factuality_eval(judge_llm_result_file: str | Path):
    file_in = Path(judge_llm_result_file)
    results_dict = load_jsonl(filepath=file_in)
    
    summarizer_lm_wise_results = {}
    full_results = {
        'gt_factuality_labels': [],
        'pred_factuality_labels': [],
        'gt_factuality_scores': [],
        'pred_factuality_scores': [],
    }

    for result in results_dict:
        summarizer_lm = result["summarizer"]
        if summarizer_lm not in summarizer_lm_wise_results:
            summarizer_lm_wise_results[summarizer_lm] = {
                'gt_factuality_labels': [],
                'pred_factuality_labels': [],
                'gt_factuality_scores': [],
                'pred_factuality_scores': [],
            }
        gt_factuality_labels, pred_factuality_labels = np.array(result["gt_labels"]["factuality_labels"]), np.array(result["pred_labels"]["factuality_labels"])

        gt_factuality_score = compute_faithfulness_percentage_score(gt_factuality_labels)
        pred_factuality_score = compute_faithfulness_percentage_score(pred_factuality_labels)

        full_results['gt_factuality_labels'].extend(gt_factuality_labels)
        full_results['pred_factuality_labels'].extend(pred_factuality_labels)
        full_results['gt_factuality_scores'].append(gt_factuality_score)
        full_results['pred_factuality_scores'].append(pred_factuality_score)

        summarizer_lm_wise_results[summarizer_lm]['gt_factuality_labels'].extend(gt_factuality_labels)
        summarizer_lm_wise_results[summarizer_lm]['pred_factuality_labels'].extend(pred_factuality_labels)
        summarizer_lm_wise_results[summarizer_lm]['gt_factuality_scores'].append(gt_factuality_score)
        summarizer_lm_wise_results[summarizer_lm]['pred_factuality_scores'].append(pred_factuality_score)

    bAcc = balancedAcc(full_results['gt_factuality_labels'], full_results['pred_factuality_labels'])
    logging.info(f"bAcc: {bAcc}")

    pearson_corr = pearsonr(full_results['gt_factuality_scores'], full_results['pred_factuality_scores'])
    spearman_corr = spearmanr(full_results['gt_factuality_scores'], full_results['pred_factuality_scores'])
    logging.info(f"pearson corr: {pearson_corr}")
    logging.info(f"spearman corr: {spearman_corr}")

    rank_corr = rank_correlation(model_wise_results=summarizer_lm_wise_results, key='factuality_scores')
    logging.info(f"rank corr: {rank_corr}")

    return {
        'bAcc': bAcc, 
        'pearson_corr': pearson_corr, 
        'spearman_corr': spearman_corr, 
        'rank_corr': rank_corr
    }

def completeness_and_conciseness_eval(judge_llm_result_file: str | Path):
    file_in = Path(judge_llm_result_file)
    results_dict = load_jsonl(filepath=file_in)
    
    summarizer_lm_wise_results = {}
    full_results = {
        'gt_completeness_scores': [],
        'pred_completeness_scores': [],
        'gt_conciseness_scores': [],
        'pred_conciseness_scores': [],
    }

    for result in results_dict:
        summarizer_lm = result["summarizer"]
        if summarizer_lm not in summarizer_lm_wise_results:
            summarizer_lm_wise_results[summarizer_lm] = {
                'gt_completeness_scores': [],
                'pred_completeness_scores': [],
                'gt_conciseness_scores': [],
                'pred_conciseness_scores': [],
            }
        
        # Human or SOTA's FineSurE Result's Keyfact<->Summary Alignment Labels
        gt_alignment_labels = result['gt_labels']['keyfact_alignment_labels']
        gt_sentence_labels = result['gt_labels']['matched_summary_labels']

        # LLM-as-a-Judge FineSurE Result's Keyfact<->Summary Alignment Labels
        pred_alignment_labels = result['pred_labels']['keyfact_alignment_labels']
        pred_sentence_labels = result['pred_labels']['matched_summary_labels']

        # calculate Completeness Percentage Score
        gt_completeness_score = compute_completeness_percentage_score(gt_alignment_labels)
        pred_completeness_score = compute_completeness_percentage_score(pred_alignment_labels)

        # calculate Conciseness Percentage Score
        gt_conciseness_score = compute_conciseness_percentage_score(gt_sentence_labels)
        pred_conciseness_score = compute_conciseness_percentage_score(pred_sentence_labels)

        full_results['gt_completeness_scores'].append(gt_completeness_score)
        full_results['pred_completeness_scores'].append(pred_completeness_score)
        full_results['gt_conciseness_scores'].append(gt_conciseness_score)
        full_results['pred_conciseness_scores'].append(pred_conciseness_score)

        summarizer_lm_wise_results[summarizer_lm]['gt_completeness_scores'].append(gt_completeness_score)
        summarizer_lm_wise_results[summarizer_lm]['pred_completeness_scores'].append(pred_completeness_score)
        summarizer_lm_wise_results[summarizer_lm]['gt_conciseness_scores'].append(gt_conciseness_score)
        summarizer_lm_wise_results[summarizer_lm]['pred_conciseness_scores'].append(pred_conciseness_score)

    # Calculate Correlations for COMPLETENESS
    pearson_corr = pearsonr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
    spearman_corr = spearmanr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
    rank_corr = rank_correlation(model_wise_results=summarizer_lm_wise_results, key='completeness_scores')
    completeness_eval_result = {
        'pearson_corr': pearson_corr, 
        'spearman_corr': spearman_corr, 
        'rank_corr': rank_corr
    }

    # Calculate Correlations for CONCISENESS
    pearson_corr = pearsonr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
    spearman_corr = spearmanr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
    rank_corr = rank_correlation(model_wise_results=summarizer_lm_wise_results, key='conciseness_scores')
    conciseness_eval_result = {
        'pearson_corr': pearson_corr, 
        'spearman_corr': spearman_corr, 
        'rank_corr': rank_corr
    }

    return completeness_eval_result, conciseness_eval_result


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

    for spec in config["judge-llm"]["spec"]:
        logging.info(f"Running FineSurE Pipeline with Judge LLM: {spec['checkpoint']}")
        judge_llm_identifier = simplify_checkpoint(spec['checkpoint'])
        file_out = Path(output_paths["finesure_pipeline"]) / f"{config['tag']}_finesure-by-judge-{judge_llm_identifier}.json"
        with open(file_out, 'w', encoding='utf-8') as f_out:
            for aggregate_sample_dict in data_aggregator(
                tag=config["tag"],
                factuality_file_path=output_paths["factuality"],
                alignment_file_path=output_paths["alignment"],
                out_path='output/finesure_pipeline_output/',
                pseudo_labeler_specs=config["pseudo-labeler"]["spec"],
                summarizer_lm_specs=config["summarizer"]["spec"]
            ):
                new_row = finesure_pipeline(
                    judge_llm_spec=spec,
                    sample=aggregate_sample_dict
                )
                f_out.write(json.dumps(new_row, ensure_ascii=False) + '\n')
                logging.info(f"Successfully saved newly predictions for sample_id: {new_row['sample_id']}")


if __name__ == "__main__":
    # main()
    factuality_eval_result = factuality_eval(judge_llm_result_file="output/finesure_pipeline_output/MTS_Dialog_Sample_finesure-by-judge-openai--gpt-4.1-mini-2025-04-14.json")
    conpleteness_eval_result, conciseness_eval_result = completeness_and_conciseness_eval(judge_llm_result_file="output/finesure_pipeline_output/MTS_Dialog_Sample_finesure-by-judge-openai--gpt-4.1-mini-2025-04-14.json")
    print(f'factuality eval result:\n{factuality_eval_result}')
    print(f'completeness eval result:\n{conpleteness_eval_result}')
    print(f'conciseness eval result:\n{conciseness_eval_result}')

"""
To Run: 
PYTHONPATH=. python -m src.finesure --config configs/config.yaml --logging info
"""
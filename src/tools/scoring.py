import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss
import logging

'''
 Score funtions
'''
def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    faithfulness = 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)  
    return faithfulness

def compute_completeness_percentage_score(pred_alignment_labels):
    # print(f"alignment_labels:\n{pred_alignment_labels}")
    completeness = sum(pred_alignment_labels) / len(pred_alignment_labels)  
    # print(f"completeness: {completeness}")
    return completeness

def compute_conciseness_percentage_score(pred_matched_summary_labels):
    conciseness = sum(pred_matched_summary_labels) / len(pred_matched_summary_labels)
    # print(f"conciseness: {conciseness}")
    return conciseness

def balancedAcc(gt, pred):
    '''
    A function to compute the balanced accuracy
    Args:
        - gt: ground truth labels
        - pred: predicted labels
    Return:
        - balanced accuracy
    '''
    ones, zeros = [], []
    for idx in range(len(gt)):
        if gt[idx] == 1.0:
            ones.append(pred[idx])
        elif gt[idx] == 0.0:
            zeros.append(pred[idx])

    error_acc = sum(ones) / len(ones)
    non_error_acc =  1.0 - sum(zeros) / len(zeros)

    return (error_acc + non_error_acc) / 2.0


def rank_correlation(model_wise_results, key, min_number=0):
    '''
    A function to compute the balanced accuracy
    Args:
        - model_wise_results: evaluation results per model in dict
        - key: evaluation dimension
        - min_number: the minimum number of examples to be included in the evaluation
    Return:
        - rank correlation with p value
    '''

    model_list =  model_wise_results.keys()

    # 각 요약모델 별
    models = []
    # (각 요약모델 별) 인간평가자 및 LLM Judge가 부여한 평균 스코어들 차례대로
    gt_errors = []
    pred_errors = []
    
    for model_name in model_list:
        models.append(model_name)
        gt_error, pred_error = np.mean(model_wise_results[model_name]['gt_' + key] if len(model_wise_results[model_name]['gt_' + key]) > 0 else 0), np.mean(model_wise_results[model_name]['pred_' + key] if len(model_wise_results[model_name]['pred_' + key]) > 0 else 0)  # Human GT, LLM Pred 각각의 평균값 사용
        # ⚠️ Empty List yielding mean() calculation failure and nan value -> temporarily avoided by giving 0 value when empty.

        if len(model_wise_results[model_name]['gt_' + key]) >= min_number:
            gt_errors.append(gt_error)
            pred_errors.append(pred_error)

    gt_errors = np.array(gt_errors)
    pred_errors = np.array(pred_errors)

    estimated_rank = ss.rankdata(pred_errors)
    human_rank = ss.rankdata(gt_errors)
    # print("models:", models)
    # print('gt ' + key + ':', gt_errors)
    # print('pred ' + key + ':', pred_errors )
    # print('gt rank ' + key + ':', human_rank)
    # print('pred rank ' + key + ':', estimated_rank)
    spearman_corr = spearmanr(estimated_rank, human_rank)

    return spearman_corr
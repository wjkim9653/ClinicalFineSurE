'''
 Score funtions
'''
def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    faithfulness = 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)  
    return faithfulness

def compute_completeness_percentage_score(pred_alignment_labels):
    completeness = sum(pred_alignment_labels) / len(pred_alignment_labels)  
    return completeness

def compute_conciseness_percentage_score(pred_sentence_line_numbers, num_sentences):
    conciseness = len(pred_sentence_line_numbers) / num_sentences
    return conciseness
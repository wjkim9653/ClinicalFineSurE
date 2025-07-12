import re
import ast
import json
import logging
ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']


def parsing_summarizer_output(output: str):
    '''
    Args:
        output: original raw output from summarizer lm
    Return:
        summary_string: parsed summary form (in str type)
        summary_list: a list of string that includes split summary in sentence level
    '''
    try:
        binarized = dict(output)
        assert("summary" in binarized.keys())
        summary_string = binarized["summary"]
    except Exception as e:
        logging.Error(f"Failed to binarize raw summary string into python dict\n{e}")
    
    try:
        summary_list = re.split(r'(?<=[.!?])\s+', summary_string)
    except Exception as e:
        logging.Error(f"Failed to split summary string into individual sentences.\n{e}")

    return summary_string, summary_list


def parsing_llm_fact_checking_output(output):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return: 
        pred_labels: the binary label for each sentence (0: no factuality error, 1: factuality error)
        pred_types: the error type of each sentence 
    '''

    try:
        start_idx = output.find('[')

        if start_idx != -1:
            end_idx = output.find(']')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            for out in output:
                category = out["category"]
                category = category.replace('\n', '').replace('[', '').replace(']', '')
                if category.lower() == "no error":
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
                pred_types.append(category)
            return pred_labels, pred_types
        
        else:
            start_idx = output.find('{')
            end_idx = output.find('}')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            category = output["category"]
            category = category.replace('\n', '').replace('[', '').replace(']', '')
            if category.lower() == "no error":
                pred_labels.append(0)
            else:
                pred_labels.append(1)
            pred_types.append(category)
            return pred_labels, pred_types
        
    except Exception as e:
        
        try:
            subseqs = output.split("category")

            def error_detection(subseq):
                detected = False
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        detected = True
                        detected_type = error_type
                if detected:
                    return 1, error_type
                else:
                    return 0, "no error"
                
            pred_labels, pred_types = [], []
            for subseq in subseqs:
                error_label, error_type = error_detection(subseq)
                pred_labels.append(error_label)
                pred_types.append(error_type)
        
            return pred_labels, pred_types
        
        except Exception as e:
            print('parsing error:', e)
            return [], []
        

def parsing_llm_keyfact_alighment_output(output):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return: 
        pred_labels: the binary label for each keyfact (0: no match, 1: match)
        matched_lines: the list of sentence line numbers that align with at least one keyfact
    '''
        
    try:
        output = output.replace('```', '')
        start_idx = output.find('[')
        output = output[start_idx:]
        output = ast.literal_eval(output)

        matched_lines = set()
        pred_labels = []

        for out in output:
            category = out["response"]

            if category.lower() == "yes":
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            
            if 'line number' in out:
                line_nums = out["line number"]

                for line_num in line_nums:
                    if type(line_num) is str:
                        line_num = line_num.replace('[', '').replace(']', '')
                    matched_lines.add(int(line_num))
        
        return pred_labels, list(matched_lines)
    
    except Exception as e:
        print(e)
        return [], []
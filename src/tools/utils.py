import csv
import json
from pathlib import Path


def simplify_checkpoint(checkpoint_path: str):
    """
    simplify the name of a given lm checkpoint 
    (main purpose is to make sure model checkpoint doesn't include '/' since it will mess up filename/directory settings when used in output file names)
    e.g.
    input:
        checkpoint_path (str): "google/gemma-3-4b-it:free"
    output:
        simple_checkpoint_name (str): "google--gemma-3-4b-it"
    """

    slash_idx = checkpoint_path.find('/')
    if slash_idx != -1:  # if there is '/' in checkpoint_path (e.g. openai/ at the start)
        checkpoint_path = checkpoint_path.replace("/", "--")
    
    colon_idx = checkpoint_path.find(':')
    if colon_idx != -1:  # if there is ':' in checkpoint_path (e.g. :free at the end)
        checkpoint_path = checkpoint_path[:colon_idx]

    return checkpoint_path

def read_csv(filepath: str | Path):
    """
    Read; Row-by-Row; the CSV file
    yields iterable rows
    """
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # csv.DictReader -> Read CSV rows as dict (can access w/ row name instead of idx)
        for row in reader:
            yield row  # yield iterable

def load_jsonl(filepath: str | Path):
    """
    Load and Serialize; a jsonl file; into a python list of dict
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                obj = json.loads(line)  # json -> dict
                data.append(obj)
    return data
def simplify_checkpoint(checkpoint_path: str):
    """
    simplify the name of a given lm checkpoint
    e.g.
    input:
        checkpoint_path (str): "google/gemma-3-4b-it:free"
    output:
        simple_checkpoint_name (str): "gemma-3-4b-it"
    """

    slash_idx = checkpoint_path.find('/')
    if slash_idx != -1:  # if there is '/' in checkpoint_path (e.g. openai/ at the start)
        checkpoint_path = checkpoint_path.replace("/", "--")
    
    colon_idx = checkpoint_path.find(':')
    if colon_idx != -1:  # if there is ':' in checkpoint_path (e.g. :free at the end)
        checkpoint_path = checkpoint_path[:colon_idx]

    return checkpoint_path
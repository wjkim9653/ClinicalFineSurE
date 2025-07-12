'''
for summary generation
'''
def get_summarization_prompt(transcript: str):
    '''
    Args:
        transctip: the original text (in str type) to summarize
    Return:
        prompt: the final input prompt
    '''
    prompt = \
'''
Text: %s

Instruction: Summarize the Text.

Provide your answer in JSON format. The answer should be a dictionary with the key “summary” containing a generated summary as a string:
{“summary”: “your summary”}

JSON Output:
''' % transcript
    
    return prompt

'''
for fact checking
'''
def get_fact_checking_prompt(input, sentences):
    
    ''' A function to define the input prompt
    Args:
        input: input document
        sentences: list of summary sentences
    Return: 
        prompt: the final input prompt
    '''

    num_sentences = str(len(sentences))
    sentences = '\n'.join(sentences)

    prompt = \
"""
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}, {"sentence": "third sentence", "reason": "your reason", "category": "entity error"},]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input, num_sentences, sentences)

    return prompt


'''
for keyfact extraction (w/ LLM)
'''
def get_keyfact_extraction_prompt(sentences: list):
    ''' A function to define the input prompt for keyfact-extraction
    Args:
        sentences: list of summary sentences
    Return:
        prompt: the final input prompt
    '''

    summary = ['[' + str(line_num + 1) + '] ' + sentence for line_num, sentence in enumerate(sentences)]
    summary = '\n'.join(summary)
    prompt =\
'''
You will be provided with a summary.
Your task is to decompose the summary into a set of "key facts".
A "key fact" is a single fact written as briefly and clearly as possible, encompassing at most 2-3 entities.

Here are nine examples of key facts to illustrate the desired level of granularity:
* Kevin Carr set off on his journey from Haytor.
* Kevin Carr set off on his journey from Dartmoor.
* Kevin Carr set off on his journey in July 2013.
* Kevin Carr is less than 24 hours away from completing his trip.
* Kevin Carr ran around the world unsupported.
* Kevin Carr ran with his tent.
* Kevin Carr is set to break the previous record.
* Kevin Carr is set to break the record by 24 hours.
* The previous record was held by an Australian.

Instruction:
First, read the summary carefully.
Second, decompose the summary into (at most 16) key facts.

Provide your answer in Json format.
The answer should be a dictionary with the key "key facts" containing the key facts as a list:
{"key facts": ["first key fact", "second key facts", "third key facts"]}

Summary:
%s
''' % (summary)
    
    return prompt


'''
for keyfact alignment
'''
def get_keyfact_alighment_prompt(keyfacts, sentences):
 
    ''' A function to define the input prompt
    Args:
        keyfacts: the list of keyfacts
        sentences: list of summary sentences
    Return: 
        prompt: the final input prompt
    '''

    summary = ['[' + str(line_num + 1) + '] ' + sentence for line_num, sentence in enumerate(sentences)]
    summary = '\n'.join(summary)
    num_key_facts = str(len(keyfacts))
    key_facts = '\n'.join(keyfacts)
    
    prompt = \
'''
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact. 

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}, {"key fact": "third key fact", "response": "Yes", "line number": [1, 2, 3]}]

Summary:
%s

%s key facts:
%s
''' % (summary, num_key_facts, key_facts)

    return prompt
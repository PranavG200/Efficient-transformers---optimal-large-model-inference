'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
def preprocess_glue(examples, tokenizer, sentence_keys, max_seq_len, pad_to_max, label_key="label"):
    """
    Preprocess the given examples using the specified tokenizer.

    Parameters:
    examples (dict): A dictionary containing the sentences and labels.
    tokenizer: The tokenizer to be used for processing.
    sentence_keys (tuple): A tuple containing keys for sentences in the examples.
    max_seq_len (int): The maximum sequence length for the tokenizer.
    pad_to_max (bool): A flag to indicate if padding is to be done to max length.
    label_key (str): The key for the label in examples. Defaults to 'label'.

    Returns:
    dict: A dictionary containing the tokenized sentences and labels.
    """
    # Tokenize the sentences
    args = (examples[sentence_keys[0]], examples.get(sentence_keys[1]))
    tokenized_result = tokenizer(
        *args,
        padding="max_length" if pad_to_max else False,
        max_length=max_seq_len,
        truncation=True,
    )

    # Add labels to the result if they exist
    if label_key in examples:
        tokenized_result["labels"] = examples[label_key]

    return tokenized_result

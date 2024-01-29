'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers.data.data_collator import DataCollatorWithPadding

import metrics
import preprocessing

def load_glue_dataset(task_name, tokenizer, is_training, max_seq_len=None, pad_to_max=False):
    """
    Load and preprocess a GLUE dataset for a given task.

    Args:
        task_name (str): The name of the GLUE task.
        tokenizer: Tokenizer object used for preprocessing.
        is_training (bool): Load training data if True, otherwise load validation data.
        max_seq_len (int, optional): Maximum sequence length for tokenization.
        pad_to_max (bool): If True, pad sequences to `max_seq_len`.

    Returns:
        A preprocessed dataset.
    """
    # Mapping from GLUE task names to their respective keys in the dataset.
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    keys = task_to_keys[task_name]

    raw_datasets = load_dataset("glue", task_name)
    preprocessed_datasets = raw_datasets.map(
        lambda examples: preprocessing.preprocess_glue(examples, tokenizer, keys, max_seq_len, pad_to_max),
        batched=True,
        load_from_cache_file=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # Select appropriate subset based on training/validation split.
    subset = "train" if is_training else "validation_matched" if task_name == "mnli" else "validation"
    return preprocessed_datasets[subset]


def create_glue_dataloader(task_name, tokenizer, is_training, batch_size=32, max_seq_len=None, pad_to_max=False):
    """
    Create a DataLoader for a GLUE dataset.

    Args:
        task_name (str): The name of the GLUE task.
        tokenizer: Tokenizer object used for preprocessing.
        is_training (bool): Create DataLoader for training data if True, otherwise for validation data.
        batch_size (int): Batch size for the DataLoader.
        max_seq_len (int, optional): Maximum sequence length for tokenization.
        pad_to_max (bool): If True, pad sequences to `max_seq_len`.

    Returns:
        A DataLoader for the specified GLUE dataset.
    """
    if max_seq_len is None:
        max_seq_len = metrics.get_max_sequence_length(task_name)

    dataset = load_glue_dataset(task_name, tokenizer, is_training, max_seq_len, pad_to_max)
    collate_fn = default_data_collator if pad_to_max else DataCollatorWithPadding(tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        collate_fn=collate_fn
    )
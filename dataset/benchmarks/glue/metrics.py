'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
TASK_ATTRIBUTES = {
    "stsb": (1, 128, 31.47, "spearmanr"),
    "mrpc": (2, 128, 53.24, "accuracy"),
    "rte": (2, 128, 64.59, "accuracy"),
    "sst2": (2, 64, 25.16, "accuracy"),
    "qqp": (2, 128, 30.55, "accuracy"),
    "qnli": (2, 128, 50.97, "accuracy"),
    "cola": (2, 64, 11.67, "matthews_correlation"),
    "mnli": (3, 128, 39.05, "accuracy"),
    "mnli-m": (3, 128, None, "accuracy"),
    "mnli-mm": (3, 128, None, "accuracy"),
}


def get_task_attributes(task_name, index):
    """
    Retrieve the attributes for a given task.

    Args:
    task_name (str): The name of the task.

    Returns:
    tuple: A tuple containing the label count, max sequence length, average sequence length (if available), and evaluation metric for the task.
    """
    if task_name in TASK_ATTRIBUTES:
        return TASK_ATTRIBUTES[task_name][index]
    else:
        raise ValueError(f"Task '{task_name}' not recognized.")


def get_label_count( task_name ):
    return get_task_attributes(task_name, 0)


def get_max_sequence_length( task_name ):
    return get_task_attributes(task_name, 1)


def get_avg_sequence_length( task_name ):
    return get_task_attributes(task_name, 2)


def get_evaluation_metric( task_name) :
    return get_task_attributes(task_name, 3)
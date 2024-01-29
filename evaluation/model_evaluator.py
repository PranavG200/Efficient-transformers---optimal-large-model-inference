'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import torch

from dataset.benchmarks.glue import create_glue_dataloader
from evaluation.glue_evaluator import GlueEvaluator

class ModelEvaluator:
    """
    A class to evaluate the accuracy of a given model on specified tasks.

    Attributes:
        model (torch.nn.Module): The neural network model to be evaluated.
        tokenizer (Tokenizer): The tokenizer to preprocess the data.
        task_name (str): The name of the task for evaluation (e.g., 'squad', 'glue').
    """

    def __init__(self, model, tokenizer, task_name):
        """
        Initializes the ModelEvaluator with a model, tokenizer, and task name.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name

    def evaluate_accuracy(self, head_mask, neuron_mask):
        """
        Evaluates the accuracy of the model on the given task.

        Args:
            head_mask (torch.Tensor): A mask to apply to the model heads.
            neuron_mask (torch.Tensor): A mask to apply to the model neurons.

        Returns:
            float: The accuracy of the model on the given task.
        """
        is_squad = "squad" in self.task_name
        test_batch_size = 32 if is_squad else 128

        if is_squad:
            accuracy = self._evaluate_squad_accuracy(head_mask, neuron_mask, test_batch_size)
        else:
            accuracy = self._evaluate_glue_accuracy(head_mask, neuron_mask, test_batch_size)

        return accuracy

    @torch.no_grad()
    def _evaluate_glue_accuracy(self, head_mask, neuron_mask, batch_size):
        """
        Private method to evaluate accuracy on the GLUE dataset.

        Args:
            head_mask (torch.Tensor): A mask to apply to the model heads.
            neuron_mask (torch.Tensor): A mask to apply to the model neurons.
            batch_size (int): The size of batches for evaluation.

        Returns:
            float: The accuracy on the GLUE dataset.
        """
        dataloader = create_glue_dataloader(
            self.task_name, self.tokenizer, is_training=False, batch_size=batch_size, pad_to_max=False)
        
        accuracy = GlueEvaluator(
            self.model, dataloader, self.task_name)
        return accuracy

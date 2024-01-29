'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import torch
from datasets import load_metric

from utils.architecture_utils import NeuronMaskApplier
from dataset.benchmarks.glue import get_target_metric_for_task

class GlueEvaluator:
    """
    Class to evaluate a model on a GLUE benchmark task.

    Attributes:
        model: The model to be evaluated.
        dataloader: DataLoader for the dataset to evaluate on.
        task_name: The name of the GLUE task.
    """

    def __init__(self, model, dataloader, task_name):
        self.model = model
        self.dataloader = dataloader
        self.task_name = task_name
        self.metric = load_metric("glue", task_name)

    @torch.no_grad()
    def evaluate(self, head_mask, neuron_mask):
        """
        Evaluate the model on the specified GLUE task.

        Args:
            head_mask: The mask to be applied on model heads.
            neuron_mask: The mask to be applied on neurons.

        Returns:
            The accuracy of the model on the specified task.
        """

        is_stsb = self.model.num_labels == 1
        self.model.eval()

        with NeuronMaskApplier(self.model, neuron_mask):
            for batch in self.dataloader:
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
                predictions = self._get_predictions(batch, head_mask, is_stsb)
                self.metric.add_batch(predictions=predictions, references=batch["labels"])

        return self._calculate_accuracy()

    def _get_predictions(self, batch, head_mask, is_stsb):
        """
        Get predictions from the model for a given batch.

        Args:
            batch: A batch of data.
            head_mask: The mask to be applied on model heads.
            is_stsb: Boolean indicating if the task is STSB.

        Returns:
            Model predictions for the batch.
        """

        outputs = self.model(head_mask=head_mask, **batch)
        if is_stsb:
            return outputs.logits.squeeze()
        return outputs.logits.argmax(dim=-1)

    def _calculate_accuracy(self):
        """
        Calculate the accuracy of the model on the specified task.

        Returns:
            The accuracy of the model.
        """

        eval_results = self.metric.compute()
        target_metric = get_target_metric_for_task(self.task_name)
        return eval_results[target_metric]
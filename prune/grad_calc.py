'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import torch
from utils.architecture_utils import apply_neuron_mask

class GradCollector:
    """
    A class for collecting gradient information.
    """

    def __init__(self, model, head_mask, neuron_mask, dataloader):
        """
        Initializes the GradCollector.

        Args:
            model (torch.nn.Module): The neural network model.
            head_mask (torch.Tensor): The head mask tensor.
            neuron_mask (torch.Tensor): The neuron mask tensor.
            dataloader (torch.utils.data.DataLoader): The data loader for batched data.
        """
        self.model = model
        self.head_mask = head_mask
        self.neuron_mask = neuron_mask
        self.dataloader = dataloader

    def collect_grads(self):
        """
        Collects gradient information.

        Returns:
            torch.Tensor: Gradient information for head mask.
            torch.Tensor: Gradient information for neuron mask.
        """
        self.head_mask.requires_grad_(True)
        self.neuron_mask.requires_grad_(True)

        handles = apply_neuron_mask(self.model, self.neuron_mask)

        self.model.eval()
        head_grads = []
        neuron_grads = []
        for batch in self.dataloader:
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)

            outputs = self.model(head_mask=self.head_mask, **batch)
            loss = outputs.loss
            loss.backward()

            head_grads.append(self.head_mask.grad.detach())
            self.head_mask.grad = None

            neuron_grads.append(self.neuron_mask.grad.detach())
            self.neuron_mask.grad = None

        for handle in handles:
            handle.remove()
        self.head_mask.requires_grad_(False)
        self.neuron_mask.requires_grad_(False)

        head_grads = torch.stack(head_grads, dim=0)
        neuron_grads = torch.stack(neuron_grads, dim=0)
        return head_grads, neuron_grads


@torch.no_grad()
def compute_fisher_info(grads):
    """
    Computes Fisher Information.

    Args:
        grads (torch.Tensor): Gradient information.

    Returns:
        torch.Tensor: Fisher Information.
    """
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info

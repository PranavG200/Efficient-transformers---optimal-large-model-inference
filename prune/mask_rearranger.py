import torch

class MaskRearranger:
    """
    A class that rearranges masks for neural network layers.
    It uses a greedy algorithm to determine which neurons/heads to prune based on gradients.
    """

    def __init__(self, mask, gradients):
        """
        Initializes the MaskRearranger with a given mask and gradients.

        :param mask: A tensor representing the mask for pruning.
        :param gradients: A tensor representing the gradients of neurons/heads.
        """
        self.mask = mask
        self.gradients = gradients

    @torch.no_grad()
    def _greedy_prune(self):
        """
        Private method that performs the greedy pruning algorithm on a single layer.

        :return: A new mask tensor after pruning.
        """
        num_unpruned = int(self.mask.sum())
        num_pruned = self.mask.shape[0] - num_unpruned
        if num_unpruned == 0 or num_pruned == 0:
            return self.mask

        grads_reshaped = self.gradients.permute(1, 0).contiguous()
        grads_squared_sum = grads_reshaped.pow(2).sum(dim=1)
        _, indices = grads_squared_sum.sort(descending=False)
        indices = indices.tolist()

        masked_indices = indices[:num_pruned]
        for index in indices[num_pruned:]:
            masked_indices.append(index)
            grad_vectors = grads_reshaped[masked_indices]
            grad_sum = grad_vectors.sum(dim=0)

            complement = grad_sum - grad_vectors
            grad_sum_length = complement.pow(2).sum(dim=1)

            removed = grad_sum_length.argmin()
            del masked_indices[removed]

        new_mask = torch.ones_like(self.mask)
        new_mask[masked_indices] = 0
        return new_mask

    def rearrange(self):
        """
        Public method to rearrange the mask for all layers.

        :return: A new mask tensor for all layers after rearrangement.
        """
        device = self.mask.device
        self.mask = self.mask.cpu()
        self.gradients = self.gradients.cpu()

        num_layers = self.mask.shape[0]
        for i in range(num_layers):
            self.mask[i] = self._greedy_prune()

        self.mask = self.mask.to(device)
        return self.mask
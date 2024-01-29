'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import torch

class ModelUtils:
    """Utility class for accessing different components of a given PyTorch model."""

    def __init__(self, model):
        self.model = model

    def get_backbone(self):
        """Returns the backbone of the model."""
        model_type = self.model.base_model_prefix
        return getattr(self.model, model_type)

    def get_encoder(self):
        """Returns the encoder component of the model."""
        return self.get_backbone().encoder

    def get_layer(self, index):
        """Returns a specific layer from the encoder."""
        return self.get_encoder().layer[index]

    def get_classifier(self):
        """Returns the classifier component of the model."""
        backbone = self.get_backbone()
        return self.model.classifier if backbone.pooler is not None else self.model.classifier.out_proj

    def get_layer_component(self, index, component):
        """Returns a specific component (MHA, FFN1, FFN2) of a specified layer."""
        layer = self.get_layer(index)
        return getattr(layer, component)


class NeuronMasker:
    """Context manager for applying neuron masks to a model during forward passes."""

    def __init__(self, model, neuron_mask):
        self.model = model
        self.neuron_mask = neuron_mask
        self.handles = []

    def __enter__(self):
        num_hidden_layers = self.neuron_mask.shape[0]
        for layer_idx in range(num_hidden_layers):
            ffn2 = ModelUtils(self.model).get_layer_component(layer_idx, 'output')
            handle = self._register_mask(ffn2, self.neuron_mask[layer_idx])
            self.handles.append(handle)

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()

    def _register_mask(self, module, mask):
        hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
        return module.register_forward_pre_hook(hook)


@torch.no_grad()
def remove_padding(hidden_states, attention_mask):
    """
    Removes padding from hidden states based on the attention mask.
    
    :param hidden_states: Tensor representing hidden states.
    :param attention_mask: Binary mask indicating non-padded elements.
    :return: Tensor of hidden states with padding removed.
    """
    mask_indices = torch.nonzero(attention_mask.view(-1), as_tuple=True)[0]
    return hidden_states.view(-1, hidden_states.shape[2]).index_select(0, mask_indices)


@torch.no_grad()
def collect_layer_inputs(model, head_mask, neuron_mask, layer_idx, prev_inputs):
    """
    Collects inputs for a specific layer of the model.

    :param model: The model to be analyzed.
    :param head_mask: Mask to be applied to the multi-head attention mechanism.
    :param neuron_mask: Mask to be applied to the neurons.
    :param layer_idx: Index of the target layer.
    :param prev_inputs: Previous inputs to the model.
    :return: List of inputs for the specified layer.
    """
    model_utils = ModelUtils(model)
    layers = [model_utils.get_layer(i) for i in range(len(model_utils.get_encoder().layer))]

    inputs = []
    if layer_idx == 0:
        handle = _hijack_input(layers[0], inputs)
        _process_batches(model, head_mask, neuron_mask, prev_inputs)
        handle.remove()
    else:
        inputs.extend(_prepare_inputs_for_layer(layers, layer_idx, prev_inputs, head_mask))

    return inputs

def _hijack_input(module, list_to_append):
    """Attaches a hook to a module to append its inputs to a list."""
    hook = lambda _, inputs: list_to_append.append(inputs)
    return module.register_forward_pre_hook(hook)

def _process_batches(model, head_mask, neuron_mask, batches):
    """Processes batches of data through the model with neuron masking."""
    with NeuronMasker(model, neuron_mask):
        for batch in batches:
            batch_cuda = {k: v.to("cuda") for k, v in batch.items()}
            model(head_mask=head_mask, **batch_cuda)

def _prepare_inputs_for_layer(layers, layer_idx, prev_inputs, head_mask):
    """Prepares inputs for a specific layer from previous layer outputs."""
    prepared_inputs = []
    for batch in prev_inputs:
        prev_output = layers[layer_idx - 1](*batch)
        batch[0] = prev_output[0]
        batch[2] = head_mask[layer_idx].view(1, -1, 1, 1)
        prepared_inputs.append(batch)
    return prepared_inputs

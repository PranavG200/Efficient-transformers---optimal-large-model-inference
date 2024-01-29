'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
from tqdm import tqdm
import torch
from utils.algebra_utils import LSMRSolver
from utils.architecture_utils import (
    extract_layers,
    extract_mha_projection,
    extract_ffn2_layer,
    hijack_model_input,
    NeuronMasking,
    remove_padding_from_tensor,
    gather_layer_inputs,
)

class ModelAnalyzer:
    """
    Class to analyze and manipulate neural network models, particularly focusing on
    attention and feed-forward neural network layers.
    """

    def __init__(self, model, config):
        """
        Initialize the ModelAnalyzer.

        :param model: The neural network model to be analyzed.
        :param config: Configuration object containing model-specific settings.
        """
        self.model = model
        self.config = config

    @torch.no_grad()
    def analyze_mha_layer(self, teacher_inputs, teacher_neuron_mask, student_inputs, 
                          student_head_mask, student_neuron_mask, layer_idx):
        """
        Analyze Multi-Head Attention (MHA) layer to understand the relationship between 
        teacher and student models in terms of attention heads.

        :param teacher_inputs: Input data for the teacher model.
        :param teacher_neuron_mask: Mask for neurons in the teacher model.
        :param student_inputs: Input data for the student model.
        :param student_head_mask: Mask for attention heads in the student model.
        :param student_neuron_mask: Mask for neurons in the student model.
        :param layer_idx: Index of the layer to be analyzed.
        :return: Tuple of matrices ATA and ATB representing the analysis results.
        """
        num_attention_heads = self.config.num_attention_heads
        hidden_size = self.config.hidden_size
        attention_head_size = int(hidden_size / num_attention_heads)

        active_heads = student_head_mask[layer_idx].nonzero().flatten()
        num_active_heads = active_heads.shape[0]

        layer = extract_layers(self.model)[layer_idx]
        mha_projection = extract_mha_projection(self.model, layer_idx)
        weights_per_head = mha_projection.dense.weight.t().view(num_attention_heads, -1, hidden_size)
        weights_per_head = weights_per_head.index_select(dim=0, index=active_heads)

        captured_inputs = []
        input_capture_handle = hijack_model_input(mha_projection, captured_inputs)

        ATA = torch.zeros(num_active_heads + 1, num_active_heads + 1).cuda()
        ATB = torch.zeros(num_active_heads + 1).cuda()

        self.model.eval()
        for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
            attention_mask = (teacher_batch[1] == 0)
            student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1)

            with NeuronMasking(self.model, teacher_neuron_mask):
                layer(*teacher_batch)
            hidden_states, input_tensor = captured_inputs.pop(0)
            teacher_output = mha_projection.dense(hidden_states) + input_tensor
            teacher_output = remove_padding_from_tensor(teacher_output, attention_mask)

            with NeuronMasking(self.model, student_neuron_mask):
                layer(*student_batch)
            hidden_states, input_tensor = captured_inputs.pop(0)
            hidden_states = remove_padding_from_tensor(hidden_states, attention_mask)
            input_tensor = remove_padding_from_tensor(input_tensor, attention_mask)

            hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
            hidden_states = hidden_states.permute(1, 0, 2)
            hidden_states = hidden_states.index_select(dim=0, index=active_heads)

            outputs_per_head = hidden_states @ weights_per_head
            outputs_per_head = outputs_per_head.view(num_active_heads, -1)

            A = outputs_per_head.t()
            A = torch.cat([A, torch.ones(A.shape[0], 1).cuda()], dim=1)
            B = teacher_output - mha_projection.dense.bias - input_tensor
            B = B.flatten()

            ATA += A.t() @ A
            ATB += A.t() @ B

        input_capture_handle.remove()
        return ATA, ATB


    @torch.no_grad()
    def analyze_ffn_layer(self, teacher_inputs, teacher_neuron_mask, student_inputs, 
                        student_head_mask, student_neuron_mask, layer_idx, cls_only=False):
        """
        Analyze Feed-Forward Network (FFN) layer to understand the relationship between 
        teacher and student models in terms of neuron activations.

        :param teacher_inputs: Input data for the teacher model.
        :param teacher_neuron_mask: Mask for neurons in the teacher model.
        :param student_inputs: Input data for the student model.
        :param student_head_mask: Mask for attention heads in the student model.
        :param student_neuron_mask: Mask for neurons in the student model.
        :param layer_idx: Index of the layer to be analyzed.
        :param cls_only: Boolean flag to indicate whether to focus only on CLS token.
        :return: Tuple of matrices ATA and ATB representing the analysis results.
        """
        layer = extract_layers(self.model)[layer_idx]
        ffn2 = extract_ffn2_layer(self.model, layer_idx)
        weights_per_neuron = ffn2.dense.weight.t()

        nonzero_neurons = student_neuron_mask[layer_idx].nonzero().flatten()
        num_neurons = nonzero_neurons.shape[0]
        weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons)
        W = weights_per_neuron @ weights_per_neuron.t()

        inputs = []
        handle = hijack_model_input(ffn2, inputs)

        ATA = torch.zeros(num_neurons, num_neurons).cuda()
        ATB = torch.zeros(num_neurons).cuda()

        self.model.eval()
        for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
            attention_mask = (teacher_batch[1] == 0)
            student_batch[2] = student_head_mask[layer_idx].view(1, -1, 1, 1)

            # Process teacher model outputs
            with NeuronMasking(self.model, teacher_neuron_mask):
                layer(*teacher_batch)
            hidden_states, input_tensor = inputs.pop(0)
            teacher_output = ffn2.dense(hidden_states) + input_tensor
            if cls_only:
                teacher_output = teacher_output[:, 0, :]
            else:
                teacher_output = remove_padding_from_tensor(teacher_output, attention_mask)

            # Process student model outputs
            with NeuronMasking(self.model, student_neuron_mask):
                layer(*student_batch)
            hidden_states, input_tensor = inputs.pop(0)
            if cls_only:
                hidden_states = hidden_states[:, 0, :]
                input_tensor = input_tensor[:, 0, :]
            else:
                hidden_states = remove_padding_from_tensor(hidden_states, attention_mask)
                input_tensor = remove_padding_from_tensor(input_tensor, attention_mask)

            hidden_states = hidden_states.t()
            hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)

            ATA += W * (hidden_states @ hidden_states.t())

            B = teacher_output - ffn2.dense.bias - input_tensor
            ATB += (hidden_states.unsqueeze(1) @ (weights_per_neuron @ B.t()).unsqueeze(2)).squeeze()

        handle.remove()
        return ATA, ATB


    @torch.no_grad()
    def rescale_masks(self, teacher_head_mask, teacher_neuron_mask, 
                    student_head_mask, student_neuron_mask, dataloader, 
                    classification_task=False):
        """
        Rescale masks for both student and teacher models to align their performance.

        :param teacher_head_mask: Head mask for the teacher model.
        :param teacher_neuron_mask: Neuron mask for the teacher model.
        :param student_head_mask: Head mask for the student model.
        :param student_neuron_mask: Neuron mask for the student model.
        :param dataloader: DataLoader object for iterating over data.
        :param classification_task: Boolean flag to indicate if the task is classification.
        :return: Tuple of rescaled head mask and neuron mask.
        """
        num_hidden_layers = self.config.num_hidden_layers
        rescaled_head_mask = student_head_mask.clone()
        rescaled_neuron_mask = student_neuron_mask.clone()

        for layer_idx in tqdm(range(num_hidden_layers)):
            teacher_inputs = gather_layer_inputs(
                self.model, teacher_head_mask, teacher_neuron_mask, layer_idx,
                prev_inputs=dataloader if layer_idx == 0 else teacher_inputs
            )
            student_inputs = gather_layer_inputs(
                self.model, rescaled_head_mask, rescaled_neuron_mask, layer_idx,
                prev_inputs=dataloader if layer_idx == 0 else student_inputs
            )

            # Rescale attention heads if needed
            if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
                ATA, ATB = self.analyze_mha_layer(
                    teacher_inputs, teacher_neuron_mask, student_inputs,
                    rescaled_head_mask, rescaled_neuron_mask, layer_idx
                )
                scale_factor, success = LSMRSolver(ATA, ATB)
                if not success:
                    break
                scale_factor = scale_factor[:-1]
                if scale_factor.max() > 10 or scale_factor.min() < -10:
                    break
                nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
                for index, scale in zip(nonzero_heads, scale_factor):
                    rescaled_head_mask[layer_idx][index] *= scale

            # Rescale neurons if needed
            if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
                cls_only = classification_task and (layer_idx == num_hidden_layers - 1)
                ATA, ATB = self.analyze_ffn_layer(
                    teacher_inputs, teacher_neuron_mask, student_inputs,
                    rescaled_head_mask, rescaled_neuron_mask, layer_idx,
                    cls_only=cls_only
                )
                scale_factor, success = LSMRSolver(ATA, ATB)
                if not success:
                    break
                if scale_factor.max() > 10 or scale_factor.min() < -10:
                    break
                nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
                for index, scale in zip(nonzero_neurons, scale_factor):
                    rescaled_neuron_mask[layer_idx][index] *= scale

        return rescaled_head_mask, rescaled_neuron_mask

'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import argparse
import math
import os

import torch
import torch.nn as nn
from transformers import AutoConfig

from utils.timer import CPUTimer, GPUTimer


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer for BERT.
    """

    def __init__(self, num_attention_heads, attention_head_size, hidden_size):
        """
        Initialize the Multi-Head Attention Layer.

        Args:
            num_attention_heads (int): Number of attention heads.
            attention_head_size (int): Size of each attention head.
            hidden_size (int): Hidden size of the model.
        """
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size

        self.all_head_size = num_attention_heads * attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.output = nn.Linear(self.all_head_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        """
        Transpose the input tensor for calculating attention scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Forward pass of the Multi-Head Attention Layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: Output hidden states after attention.
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.output(context_layer) + hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network Layer for BERT.
    """

    def __init__(self, hidden_size, intermediate_size):
        """
        Initialize the Feed-Forward Network Layer.

        Args:
            hidden_size (int): Hidden size of the model.
            intermediate_size (int): Size of the intermediate layer.
        """
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        """
        Forward pass of the Feed-Forward Network Layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: Output hidden states after feed-forward processing.
        """
        ffn1_output = self.ffn1(hidden_states)
        ffn1_output = self.gelu(ffn1_output)
        ffn2_output = self.ffn2(ffn1_output)
        hidden_states = ffn2_output + hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


@torch.no_grad()
def calculate_mha_latency(config, device, input_shape, num_warmup=100, num_iter=100):
    """
    Calculate the latency of the Multi-Head Attention layer.

    Args:
        config (AutoConfig): BERT model configuration.
        device (str): Device ('cpu' or 'cuda') for evaluation.
        input_shape (tuple): Shape of the input tensor.
        num_warmup (int): Number of warm-up iterations.
        num_iter (int): Number of iterations for latency calculation.

    Returns:
        list: List of mean latencies for different numbers of attention heads.
    """
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    latencies = []
    for num_heads in range(1, num_attention_heads + 1):
        model = MultiHeadAttentionLayer(num_heads, attention_head_size, hidden_size)
        model = model.to(device).eval()

        x = torch.randn(input_shape).to(device)

        for _ in range(num_warmup):
            model(x)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)
        for _ in range(num_iter):
            with timer:
                model(x)

        mean_latency = sum(timer.timelogs) / num_iter
        latencies.append(mean_latency)
    return latencies


@torch.no_grad()
def calculate_ffn_latency(config, device, input_shape, num_warmup=10, num_iter=10):
    """
    Calculate the latency of the Feed-Forward Network layer.

    Args:
        config (AutoConfig): BERT model configuration.
        device (str): Device ('cpu' or 'cuda') for evaluation.
        input_shape (tuple): Shape of the input tensor.
        num_warmup (int): Number of warm-up iterations.
        num_iter (int): Number of iterations for latency calculation.

    Returns:
        list: List of mean latencies for different numbers of intermediate neurons.
    """
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    latencies = []
    for num_neurons in range(1, intermediate_size + 1):
        model = FeedForwardNetwork(hidden_size, num_neurons)
        model = model.to(device).eval()

        x = torch.randn(input_shape).to(device)

        for _ in range(num_warmup):
            model(x)

        timelogs = []
        timer = CPUTimer(timelogs) if device == "cpu" else GPUTimer(timelogs)

        for _ in range(num_iter):
            with timer:
                model(x)

        mean_latency = sum(timer.timelogs) / num_iter
        latencies.append(mean_latency)
    return latencies
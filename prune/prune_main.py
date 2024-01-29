'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from dataset.benchmarks.glue import glue_dataset, max_seq_length, avg_seq_length

from evaluation import *
from prune import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/bert-base-uncased/mrpc")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--metric", type=str, default="mac", choices=["mac", "latency"])
    parser.add_argument("--constraint", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mha_lut", type=str, default="lut/mha_lut.pt")
    parser.add_argument("--ffn_lut", type=str, default="lut/ffn_lut.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.ckpt_dir)
    model_generator = AutoModelForSequenceClassification
    model = model_generator.from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        use_auth_token=None,
    )

    training_dataset = glue_dataset(
        args.task_name,
        tokenizer,
        training=True,
        max_seq_len=max_seq_length(args.task_name),
        pad_to_max=False,
    )

    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )

    sample_batch_size = 32
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()

    start = time.time()
    # Search the optimal mask
    head_grads, neuron_grads = GradCollector.collect_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    teacher_constraint = PruningScheduler.generate_schedule(target=args.constraint, num_iter=2)[0]
    if args.metric == "mac":
        teacher_head_mask, teacher_neuron_mask = PerformanceTracker.search_mac(
            config,
            head_grads,
            neuron_grads,
            128,
            teacher_constraint,
        )
        head_mask, neuron_mask = PerformanceTracker.search_mac(
            config,
            head_grads,
            neuron_grads,
            128,
            args.constraint,
        )
        pruned_mac, orig_mac = PerformanceTracker.search_mac(head_mask, neuron_mask, 128, config.hidden_size)
        print(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    elif args.metric == "latency":
        mha_lut = torch.load(args.mha_lut)
        ffn_lut = torch.load(args.ffn_lut)
        teacher_head_mask, teacher_neuron_mask = PerformanceTracker.search_latency(
            config,
            head_grads,
            neuron_grads,
            teacher_constraint,
            mha_lut,
            ffn_lut,
        )
        head_mask, neuron_mask = PerformanceTracker.search_latency(
            config,
            head_grads,
            neuron_grads,
            args.constraint,
            mha_lut,
            ffn_lut,
        )
        pruned_latency = PerformanceTracker.search_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
        print(f"Pruned Model Latency: {pruned_latency:.2f} ms")

    # Rearrange the mask
    head_mask = MaskRearranger.rearrange(head_mask, head_grads)
    neuron_mask = MaskRearranger.rearrange(neuron_mask, neuron_grads)

    # Rescale the mask by solving a least squares problem
    head_mask, neuron_mask = ModelAnalyzer.rescale_masks(
        model,
        config,
        teacher_head_mask,
        teacher_neuron_mask,
        head_mask,
        neuron_mask,
        sample_dataloader,
        classification_task=True,
    )

    end = time.time()
    print(f"{args.task_name} Pruning time (s): {end - start}")

    # Evaluate the accuracy
    test_acc = ModelEvaluator.evaluate_accuracy(model, head_mask, neuron_mask, tokenizer, args.task_name)
    print(f"{args.task_name} Test accuracy: {test_acc:.4f}")

    # Save the masks
    torch.save(head_mask, os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.output_dir, "neuron_mask.pt"))
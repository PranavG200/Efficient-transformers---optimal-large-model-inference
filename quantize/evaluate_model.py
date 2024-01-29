'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from utils import load_and_cache_examples, compute_metrics

class Evaluator:
    def __init__(self, args, model, tokenizer):
        """
        Initializes the Evaluator.

        Args:
            args (argparse.Namespace): Command-line arguments.
            model (torch.nn.Module): The model to evaluate.
            tokenizer: The tokenizer for processing inputs.
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, prefix=""):
        """
        Evaluates the model on the specified task(s).

        Args:
            prefix (str): A prefix for saving evaluation results.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        eval_task_names = ("mnli", "mnli-mm") if self.args.task_name == "mnli" else (self.args.task_name,)
        eval_outputs_dirs = (self.args.output_dir, self.args.output_dir + '-MM') if self.args.task_name == "mnli" else (self.args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = load_and_cache_examples(self.args, eval_task, self.tokenizer, evaluate=True)

            if not os.path.exists(eval_output_dir) and self.args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

            # Multi-GPU evaluation
            if self.args.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)

            # Evaluation
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", self.args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.model.eval()
                batch = tuple(t.to(self.args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[3]
                    }
                    if self.args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if self.args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif self.args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results
'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import time

class ModelEvaluatorTime:
    """
    A class for evaluating a machine learning model.

    Args:
        model: The machine learning model to evaluate.
        configs: Configuration settings for evaluation.
        tokenizer: Tokenizer used for preprocessing text data.
    """

    def __init__(self, model, configs, tokenizer):
        self.model = model
        self.configs = configs
        self.tokenizer = tokenizer

    def evaluate_model(self):
        """
        Evaluate the model's performance and print the results.

        Returns:
            result: The evaluation result.
        """
        eval_start_time = time.time()
        result = self.evaluate()
        eval_end_time = time.time()
        eval_duration_time = eval_end_time - eval_start_time
        self.print_evaluation_result(result, eval_duration_time)

    def print_evaluation_result(self, result, eval_duration_time):
        """
        Print the evaluation result and the time taken for evaluation.

        Args:
            result: The evaluation result.
            eval_duration_time: The time taken for evaluation in seconds.
        """
        print(result)
        print(f"Evaluate total time (seconds): {eval_duration_time:.1f}")
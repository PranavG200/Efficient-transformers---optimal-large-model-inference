'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import os
import torch

class ModelSizeCalculator:
    """
    A utility class for calculating and printing the size of a PyTorch model in megabytes (MB).
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_and_print_model_size(model):
        """
        Calculate and print the size of the provided PyTorch model in megabytes (MB).

        Args:
            model (torch.nn.Module): The PyTorch model for which the size needs to be calculated and printed.
        """
        try:
            # Save the model's state dictionary to a temporary file
            torch.save(model.state_dict(), "temp_model_state_dict.pth")

            # Calculate and print the size of the temporary file in MB
            size_in_mb = os.path.getsize("temp_model_state_dict.pth") / 1e6
            print(f"Model Size (MB): {size_in_mb:.2f} MB")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Remove the temporary file
            os.remove("temp_model_state_dict.pth")
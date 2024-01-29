'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import math

class PruningScheduler:
    """
    A class for generating pruning schedules.

    Attributes:
        target (float): The target pruning rate.
        num_iterations (int): The number of iterations to generate the schedule for.
    """

    def __init__(self, target, num_iterations):
        """
        Initializes the PruningScheduler with the target pruning rate and the number of iterations.

        Args:
            target (float): The target pruning rate (0.0 to 1.0).
            num_iterations (int): The number of iterations to generate the schedule for.
        """
        self.target = target
        self.num_iterations = num_iterations

    def generate_schedule(self):
        """
        Generates a pruning schedule.

        Returns:
            list: A list of pruning rates for each iteration.
        """
        pruning_factor = math.pow(self.target, 1 / self.num_iterations)
        schedule = [pruning_factor ** i for i in range(1, self.num_iterations)] + [self.target]
        return schedule

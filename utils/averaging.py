'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
class RunningAverage:
    """
    A class for calculating and maintaining a running average of values.

    Attributes:
        name (str): The name of the average.
        value (float): The current value.
        average (float): The running average.
        total_sum (float): The sum of all values.
        count (int): The number of values added.
    """

    def __init__(self, name):
        """
        Initialize a RunningAverage object.

        Args:
            name (str): The name of the average.
        """
        self.name = name
        self.reset()

    def reset(self):
        """
        Reset the average to initial values.
        """
        self.value = 0.0
        self.average = 0.0
        self.total_sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        """
        Update the running average with a new value.

        Args:
            value (float): The new value to be added.
            count (int): The number of times the value should be added (default is 1).
        """
        self.value = value
        self.total_sum += value * count
        self.count += count
        self.average = self.total_sum / self.count

    def __str__(self):
        """
        Return a string representation of the RunningAverage object.
        """
        return f"{self.name}: {self.average:.4f}"
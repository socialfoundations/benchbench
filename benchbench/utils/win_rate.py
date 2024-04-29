import math
import numpy as np


class WinningRate:
    def __init__(self, data, cols):
        """
        Calculate the winning rate of a list of models.

        Args:
            data (pd.DataFrame): Each row represents a model, each column represents a task.
            cols (list): The column names of the tasks.

        Returns:
            None
        """
        m = len(data)
        n = len(cols)
        self.win_rate = np.zeros([m, m])
        data = data[cols].values
        for i in range(m):
            for j in range(m):
                n_win, n_tot = 0, 0
                for k in range(n):
                    if not math.isnan(data[i, k]) and not math.isnan(data[j, k]):
                        n_tot += 1
                        if float(data[i, k]) > float(data[j, k]) and i != j:
                            n_win += 1
                self.win_rate[i, j] = n_win / n_tot if n_tot > 0 else 0

    def get_winning_rate(self, model_indices=None):
        """
        Get the winning rate of the selected models.

        Args:
            model_indices (list): Indices of the selected models.

        Returns:
            float: The winning rate.
        """
        model_indices = (
            np.arange(len(self.win_rate)) if model_indices is None else model_indices
        )
        return self.win_rate[model_indices][:, model_indices].mean(axis=1)

############################################################################################
#                                                                                          #
#                                    Some utility functions                                #
#                                                                                          #
#                                   Lionel Cheng, 01.06.2022                               #
#                                                                                          #
############################################################################################

import pandas as pd

class MetricTracker:
    """
    Track metrics from the network by storing them in a pandas.DataFrame and sending them to TensorBoard if a writer
    is specified (deactivated functionality)
    """
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """ Reset all keys to zero. """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """ Update key in DataFrame with the given value and a count if specified. """
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """ Return the average for the given key. """
        return self._data.average[key]

    def result(self):
        """ Return averages as a dictionary. """
        return dict(self._data.average)


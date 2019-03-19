import numpy as np


def get_first_capacitor(event, nr_module):
    """
    Get first capacitor values from event for nr module.
    Parameters
    ----------
    event : `ctapipe` event-container
    nr_module : number of module
    """
    fc = np.zeros((2, 7))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr_module * 8:
                                                        (nr_module + 1) * 8]
    high_gain = 0
    low_gain = 1
    # First capacitor order according Dragon v5 board data format
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[high_gain, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[low_gain, i] = first_cap[j]
    return fc
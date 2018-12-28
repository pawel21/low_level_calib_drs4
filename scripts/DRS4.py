class DRS4:
    def __init__(self):
        pass

    @staticmethod
    def get_first_capacitor_for_module(event, nr):
        hg = 0
        lg = 1
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[hg, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[lg, i] = first_cap[j]
        return fc


def get_first_capacitor_array(event):
    return event.lst.tel[0].evt.first_capacitor_id[:]

import numpy as np
import os
import struct


class DragonPedestal:
    n_pixels = 7
    roisize = 40
    size4drs = 4*1024
    high_gain = 0
    low_gain = 1

    def __init__(self):
        self.first_capacitor = np.zeros((2, 8))
        self.meanped = np.zeros((2, self.n_pixels, self.size4drs))
        self.numped = np.zeros((2, self.n_pixels, self.size4drs))
        self.rms = np.zeros((2, self.n_pixels, self.size4drs))

    def fill_pedestal_event(self, event, nr):
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            self.first_capacitor[self.high_gain, i] = first_cap[j]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            self.first_capacitor[self.low_gain, i] = first_cap[j]

        waveform = event.r0.tel[0].waveform[:, nr * 7:(nr + 1) * 7, :]
        for i in range(0, 2):
            for j in range(0, self.n_pixels):
                fc = int(self.first_capacitor[i, j])
                posads0 = int((2+fc)%self.size4drs)
                if posads0 + 40 < 4096:
                    self.meanped[i, j, (posads0+2):(posads0+38)] += waveform[i, j, 2:38]
                    self.numped[i, j, (posads0 + 2):(posads0 + 38)] += 1
                else:
                    for k in range(2, self.roisize-2):
                        posads = int((k+fc)%self.size4drs)
                        val = waveform[i, j, k]
                        self.meanped[i, j, posads] += val
                        self.numped[i, j, posads] += 1

    def finalize_pedestal(self):
        try:
            self.meanped = self.meanped/self.numped
            self.rms = self.rms/self.numped
        except Exception as err:
            print(err)


def get_first_capacitor(event, nr):
    hg = 0
    lg = 1
    fc = np.zeros((2, 8))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[hg, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[lg, i] = first_cap[j]
    return fc


def write_pedestal_to_file(file_name, PedList, number_modules):
    f_out = open(file_name, 'wb')
    # header
    f_out.write(struct.pack('>B', 1))  # version 1
    f_out.write(struct.pack('>H', 7))  # number of pixels
    f_out.write(struct.pack('>H', 4096))  # number of samples
    f_out.write(struct.pack('>H', 40))  # RoI
    f_out.write(struct.pack('>H', number_modules))  # number of modules

    for nr in range(0, number_modules):
        # high gain
        for pixel in range(0, 7):
            pedestal_value = (PedList[nr].meanped[0, pixel, :])
            for value in (pedestal_value):
                if np.isnan(value):
                    value = 300 # offset if not enough value to calculate baseline
                    f_out.write(struct.pack('>H', int(value)))
                else:
                    f_out.write(struct.pack('>H', int(value)))
        # low gain
        for pixel in range(0, 7):
            pedestal_value = (PedList[nr].meanped[1, pixel, :])
            for value in (pedestal_value):
                if np.isnan(value):
                    value = 300 # offset if not enough value to calculate baseline
                    f_out.write(struct.pack('>H', int(value)))
                else:
                    f_out.write(struct.pack('>H', int(value)))

    f_out.close()
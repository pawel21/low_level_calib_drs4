import argparse
import numpy as np
import os
import sys
from astropy.io import fits
from numba import jit, prange

from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker


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

    def fill_pedestal_event(self, event, nr):
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            self.first_capacitor[self.high_gain, i] = first_cap[j]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            self.first_capacitor[self.low_gain, i] = first_cap[j]

        waveform = event.r0.tel[0].waveform[:, nr * 7:(nr + 1) * 7, :]
        self._fill_pedestal_event_jit(waveform, self.first_capacitor, self.meanped, self.numped)

    @staticmethod
    @jit
    def _fill_pedestal_event_jit(waveform, first_cap, meanped, numped):
        size4drs = 4096
        roisize = 40
        for i in range(0, 2):
            for j in range(0, 7):
                fc = int(first_cap[i, j])
                posads0 = int((2+fc)%size4drs)
                if posads0 + 40 < 4096:
                    meanped[i, j, posads0:(posads0+36)] += waveform[i, j, 2:38]
                    numped[i, j, posads0:(posads0 + 36)] += 1
                else:
                    for k in range(2, roisize-2):
                        posads = int((k+fc)%size4drs)
                        val = waveform[i, j, k]
                        meanped[i, j, posads] += val
                        numped[i, j, posads] += 1

    def finalize_pedestal(self):
        try:
            self.meanped = self.meanped/self.numped
        except Exception as err:
            print(err)
            print("Not enough events to coverage all capacitor. "
                  "Please use more file to create pedestal file.")


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


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="Path to fitz.fz file to create pedestal file. Can be one path or more",
                    nargs='+', type=str)
parser.add_argument("--output_file", help="Path where script should create pedestal file",
                    nargs=1, type=str)
args = parser.parse_args()

print("Input file: ", args.input_file)
print("Output file: ", args.output_file)

reader = LSTEventSource(input_url=args.input_file[0])
seeker = EventSeeker(reader)
ev = seeker[0]
n_modules = ev.lst.tel[0].svc.num_modules

ped = DragonPedestal()
PedList = []
pedestal_value_array = np.zeros((n_modules, 2, 7, 4096), dtype=np.uint16)

# create list of Pedestal class for each module
for i in range(0, n_modules):
    PedList.append(DragonPedestal())

using_files_name = ''
for file_path in args.input_file:
    using_files_name += (file_path.split("/")[-1])
    using_files_name += ", "
    input_file_reader = LSTEventSource(input_url=file_path)
    for event in input_file_reader:
        for nr_module in range(0, n_modules):
            PedList[nr_module].fill_pedestal_event(event, nr=nr_module)


# Finalize pedestal and write to fits fiel
for i in range(0, n_modules):
    PedList[i].finalize_pedestal()
    PedList[i].meanped[np.isnan(PedList[i].meanped)] = 300 # fill 300 where occurs NaN value
    pedestal_value_array[i, :, :, :] = PedList[i].meanped

hdu = fits.PrimaryHDU()
hdu.data = pedestal_value_array
hdu.header["DATA"] = using_files_name
hdu.writeto(args.output_file[0])


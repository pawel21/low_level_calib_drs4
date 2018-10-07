import numpy as np
import os
import sys
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker

from create_binary_file_with_pedestals_functions import DragonPedestal, write_pedestal_to_file

path_to_data = sys.argv[1]
output_path = sys.argv[2]

reader = LSTEventSource(input_url=path_to_data)
seeker = EventSeeker(reader)
ev = seeker[0]
number_modules = ev.lst.tel[0].svc.num_modules

ped = DragonPedestal()
PedList = []

inputfile_reader_event = LSTEventSource(input_url=path_to_data)

# create list of Pedestal class
for i in range(0, number_modules):
    PedList.append(DragonPedestal())

for nr_event, event in enumerate(inputfile_reader_event):
    print(nr_event)
    for nr_module in range(0, number_modules):
        PedList[nr_module].fill_pedestal_event(event, nr=nr_module)

for i in range(0, number_modules):
    PedList[i].finalize_pedestal()


write_pedestal_to_file(output_path, PedList, number_modules)

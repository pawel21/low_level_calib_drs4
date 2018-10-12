from abc import abstractmethod
import numpy as np

from numba import jit, njit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Unicode

class CameraR1Calibrator(Component):
    """
    The base R1-level calibrator. Fills the r1 container.
    The R1 calibrator performs the camera-specific R1 calibration that is
    usually performed on the raw data by the camera server. This calibrator
    exists in ctapipe for testing and prototyping purposes.
    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the r1 calibrators. Fills the r1 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)
        self._r0_empty_warn = False

    @abstractmethod
    def calibrate(self, event):
        """
        Abstract method to be defined in child class.
        Perform the conversion from raw R0 data to R1 data
        (ADC Samples -> PE Samples), and fill the r1 container.
        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """

    def check_r0_exists(self, event, telid):
        """
        Check that r0 data exists. If it does not, then do not change r1.
        This ensures that if the containers were filled from a file containing
        r0 data, it is not overwritten by non-existant data.
        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.
        Returns
        -------
        bool
            True if r0.tel[telid].waveform is not None, else false.
        """
        r0 = event.r0.tel[telid].waveform
        if r0 is not None:
            return True
        else:
            if not self._r0_empty_warn:
                self.log.warning("Encountered an event with no R0 data. "
                                 "R1 is unchanged in this circumstance.")
                self._r0_empty_warn = True
        return False


class LSTR1Calibrator(CameraR1Calibrator):

    pedestal_path = Unicode(
        '',
        allow_none=True,
        help='Path to the LST pedestal file'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        The R1 calibrator for LST data.
        Fills the r1 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)
        self.telid = 0
        self.pedestal_value_array = None
        self.n_pixels = 7
        self.size4drs = 4 * 1024
        self.roisize = 40
        self.offset = 300
        self.high_gain = 0
        self.low_gain = 1

        self._load_calib()

        self.gain_range = range(0, 2)
        self.pixel_range = range(0, 7)
        self.roi_range = range(0, 40)
    def calibrate(self, event):
        """
        Perform calibration on event using pedestal file.
        Parameters
        ----------
        event : `ctapipe` event-container
        """
        event.r1.tel[self.telid].waveform = np.zeros(event.r0.tel[self.telid].waveform.shape, dtype=np.uint16)
        fc_all = np.zeros((self.number_of_modules_from_file, 2, 7))

        for nr_module in range(0, self.number_of_modules_from_file):
            fc_all[nr_module, :, :] = self._get_first_capacitor(event, nr_module)
     #       first_cap = self._get_first_capacitor(event, nr_module)
     #       for gain in self.gain_range:
     #           for pixel in self.pixel_range:
     #               position = int((first_cap[gain, pixel]) % self.size4drs)
     #               event.r1.tel[self.telid].waveform[gain, pixel + nr_module * 7, :] = \
     #                   (event.r0.tel[self.telid].waveform[gain, pixel + nr_module * 7, :] -
     #                    self.pedestal_value_array[nr_module, gain, pixel, position:position+40])

        event.r1.tel[self.telid].waveform[:, :, :] = self.calibrate_jit(event.r0.tel[self.telid].waveform, fc_all,
                                                                        self.pedestal_value_array,
                                                                        self.number_of_modules_from_file)

    @staticmethod
    @njit(parallel=True)
    def calibrate_jit(event_waveform, fc_cap, pedestal_value_array, nr_clus):
        ev_waveform = np.zeros(event_waveform.shape)
        size4drs = 4096
        #first_cap = np.zeros((2, 7))
        for nr in prange(0, nr_clus):
            for gain in prange(0, 2):
                for pixel in prange(0, 7):
                    position = int((fc_cap[nr, gain, pixel]) % size4drs)
                    ev_waveform[gain, pixel + nr * 7, :] = \
                        (event_waveform[gain, pixel + nr * 7, :] -
                         pedestal_value_array[nr, gain, pixel, position:position+40])
        return ev_waveform

    def _load_calib(self):
        """
        If a pedestal file has been supplied, create a array with
        pedestal value . If it hasn't then point calibrate to
        fake_calibrate, where nothing is done to the waveform.
        """

        if self.pedestal_path:
            with open(self.pedestal_path, "rb") as binary_file:
                data = binary_file.read()
                file_version = int.from_bytes(data[0:1], byteorder='big')
                self.number_of_modules_from_file = int.from_bytes(data[7:9],
                                                                  byteorder='big')
                self.pedestal_value_array = np.zeros((self.number_of_modules_from_file, 2,
                                                      self.n_pixels, self.size4drs+40))
                self.log.info("Load binary file with pedestal version {}: {} ".format(
                    file_version, self.pedestal_path))
                self.log.info("Number of modules in file: {}".format(
                    self.number_of_modules_from_file))

                start_byte = 9
                for i in range(0, self.number_of_modules_from_file):
                    for gain in range(0, 2):
                        for pixel in range(0, self.n_pixels):
                            for cap in range(0, self.size4drs):
                                value = int.from_bytes(data[start_byte:start_byte + 2],
                                                       byteorder='big') -300
                                self.pedestal_value_array[i, gain, pixel, cap] = value
                                start_byte += 2
                            self.pedestal_value_array[i, gain, pixel, self.size4drs:self.size4drs+40] = self.pedestal_value_array[i, gain, pixel, 0:40]
        else:
            self.log.warning("No pedestal path supplied, "
                             "r1 samples will equal r0 samples.")
            self.calibrate = self.fake_calibrate

    def _get_first_capacitor(self, event, nr_module):
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
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

    def fake_calibrate(self, event):
        """
        Don't perform any calibration on the waveforms, just fill the
        R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        """

        for telid in event.r0.tels_with_data:
            if self.check_r0_exists(event, telid):
                samples = event.r0.tel[telid].waveform
                event.r1.tel[telid].waveform = samples.astype('float32')


def interpolate_spike_A(event, gain, pos, pixel, nr_clus):
    samples = event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, :]
    a = int(samples[pos-1])
    b = int(samples[pos+2])
    value1 = samples[pos - 1] + (0.33 * (b-a))
    value2 = samples[pos - 1] + (0.66 * (b-a))
    event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, pos] = value1
    event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, pos+1] = value2

def interpolate_spike_B(event, gain, spike_b_pos, pixel, nr_clus):
    samples = event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, :]
    value = 0.5 * (samples[spike_b_pos - 1] + samples[spike_b_pos + 1])
    event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, spike_b_pos] = value

def time_corr():
    pass

def int64(x):
    return x[0] + x[1] * 256 + x[2] * 256 ** 2 + x[3] * 256 ** 3 + x[4] * 256 ** 4 + x[5] * 256 ** 5 + x[
            6] * 256 ** 6 + x[7] * 256 ** 7

def ped_time(timediff):
    return 29.3 * np.power(timediff, -0.2262) - 12.4

def get_first_capacitor(event, nr):
    hg = 0
    lg = 1
    fc = np.zeros((2, 7))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[hg, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[lg, i] = first_cap[j]
    return fc
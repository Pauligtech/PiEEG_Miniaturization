from moabb.datasets import Cho2017

"""GigaDb Motor imagery dataset."""

import logging
import time

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)
GIGA_URL = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/"


class NormCho2017(BaseDataset):
    """This is the Cho2017 dataset, but with the only the first 108 trials"""

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 53)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="NormCho2017",
            interval=[0, 3],  # full trial is 0-3s, but edge effects
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fname = self.data_path(subject)

        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["eeg"]

        # fmt: off
        eeg_ch_names = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]
        # fmt: on
        emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
        ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
        ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
        montage = make_standard_montage("standard_1005")
        imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
        imagery_right = data.imagery_right - data.imagery_right.mean(
            axis=1, keepdims=True
        )

        eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack(
            [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
        )

        # ##############################  NEW CODE ########################################
        # count1 = 0
        # count2 = 0
        # # Cut the first 108 trials
        # for i in range(len(eeg_data[0])):
        #     # Count the times the stim channel is 1 and 2
        #     if eeg_data[-1][i] == 1:
        #         count1 += 1
        #         if count1 > 54:
        #             eeg_data[-1][i] = 0

        #     elif eeg_data[-1][i] == 2:
        #         count2 += 1
        #         if count2 > 54:
        #             eeg_data[-1][i] = 0

        # Check that it worked by printing the counts of 1 and 2
        # count the number of times 1 appears in the stim channel
        # postcount1 = 0
        # postcount2 = 0
        # for i in range(len(eeg_data[0])):
        #     if eeg_data[-1][i] == 1:
        #         postcount1 += 1
        #     elif eeg_data[-1][i] == 2:
        #         postcount2 += 1

        # print("Count1: ", postcount1)
        # print("Count2: ", postcount2)

        ##################################################################################

        # If count1 is

        # log.warning(
        #     "Trials demeaned and stacked with zero buffer to create "
        #     "continuous data -- edge effects present"
        # )

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        raw.set_montage(montage)

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}s{:02d}.mat".format(GIGA_URL, subject)
        return dl.data_dl(url, "GIGADB", path, force_update, verbose)

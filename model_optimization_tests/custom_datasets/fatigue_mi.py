"""Fatigue MI Dataset."""

import sys
from mne.io import Raw
from mne.io import RawArray
from mne import create_info
from pathlib import Path
import pandas as pd

from moabb.datasets.base import BaseDataset


class FatigueMI(BaseDataset):
    """Joanna fatigue MI dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ============  ===============  ===========
        Name      #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ============  ===============  ===========
        FatigueMI    32       20           2                 54  2s            300Hz                      1
        ======  =======  =======  ==========  =================  ============  ===============  ===========

    Motor imagery dataset from the Master's work of Joanna Keough.

    This Dataset contains EEG recordings from  subjects, performing a
    motor imagination task (right hand, left hand). Data have been recorded at
    300Hz with 20 dry active electrodes (A2, C3, C4, Cz, F3, F4, F7, F8, Fp1, Fp2,
    Fz, O1, O2, P3, P4, Pz, T3, T4, T5, T6) with a DSI-24C.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a left hand trial is denoted
    by 0, the start of a right hand trial by 1. Following the labelled trials above,
    the BCI was used for a cursor control task for *** minutes. The cursor
    direction was self-selected, so these trials are labelled as -1.

    The duration of each trial is 2 seconds. There are 54 trials of each class.

    references
    ----------

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 33)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            # events=dict(left_hand=1, right_hand=2, unlabelled=3),
            code="FatigueMI",
            interval=[0, 2],
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raw = Raw(self.data_path(subject), preload=True, verbose="ERROR")
        # info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # Link the home directory path
        current_path = Path().absolute()
        home_path = current_path
        sys.path.append(str(home_path))

        # Load our data, test with fatigue
        data_table_path = home_path / "dataTables"
        mi_dt = pd.read_csv(data_table_path / "fatigue_data_table.csv")

        # Remove subjects P2, P9 and P19 from the data table
        mi_dt = mi_dt[mi_dt["ParticipantLabel"] != "sub-P02"]
        mi_dt = mi_dt[mi_dt["ParticipantLabel"] != "sub-P05"]
        mi_dt = mi_dt[mi_dt["ParticipantLabel"] != "sub-P09"]
        mi_dt = mi_dt[mi_dt["ParticipantLabel"] != "sub-P19"]

        mi_dt = mi_dt[mi_dt["Task"] == "MI-S"]

        mi_dt.reset_index(inplace=True, drop=True)

        return (
            Path("~/mne_data")
            / "MNE-fatigue-mi"
            / "{}.raw.fif".format(mi_dt["ParticipantLabel"][subject - 1])
        )

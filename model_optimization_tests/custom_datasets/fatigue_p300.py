"""Fatigue MI Dataset."""

import sys
from mne.io import Raw
from pathlib import Path
import pandas as pd

from moabb.datasets.base import BaseDataset


class FatigueP300(BaseDataset):
    """Joanna fatigue P300 dataset.

    P300 dataset from the Master's work of Joanna Keough.

    This Dataset contains EEG recordings from  subjects, performing a
    motor imagination task (right hand, left hand). Data have been recorded at
    300Hz with 19 dry active electrodes (A2, C3, C4, Cz, F3, F4, F7, F8, Fp1, Fp2, 
    Fz, O1, O2, P3, P4, Pz, T3, T4, T5, T6) with a DSI-24C.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the P300 stimulus. 1 denotes a target, 2 denotes a non-target.

    references
    ----------

    """


    def __init__(self):
        super().__init__(
            subjects=list(range(1, 33)),
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=2),
            code="FatigueP300",
            interval=[0, 0.8],
            paradigm="p300",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raw = Raw(self.data_path(subject), preload=True, verbose="ERROR")
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
        p300_dt = pd.read_csv(data_table_path / "fatigue_data_table.csv")

        p300_dt = p300_dt[p300_dt["Task"] == "P300-S"]

        #Reindex the dataframe
        p300_dt.reset_index(inplace=True, drop=True)
        
        return Path("~/mne_data") / "MNE-fatigue-p300" / "{}.raw.fif". format(p300_dt["ParticipantLabel"][subject-1])
"""Optimization Gamified MI Dataset."""

import sys
from mne.io import Raw
from pathlib import Path
import pandas as pd

from moabb.datasets.base import BaseDataset


class OptGameMI(BaseDataset):
    """Dion optimization standard MI dataset.

    .. admonition:: Dataset summary


        ======  =======  =======  ==========  =================  ============  ===============  ===========
        Name      #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ======  =======  =======  ==========  =================  ============  ===============  ===========
        OptStdMI    31       16           2                 54  1.5s            256Hz                    1
        ======  =======  =======  ==========  =================  ============  ===============  ===========

    Motor imagery dataset from the PhD work of Dr. Dion Kelly.

    This Dataset contains EEG recordings from  subjects, performing a
    motor imagination task (right hand, left hand). Data have been recorded at
    256Hz with 16 gtec scarabeo active gel electrodes and gUSB amp amplifier 
    (C1, C2, C3, C4, C5, C6, CP1, CP2, CP3, CP4, CPz, Cz, FC3, FC4, FCz, Pz).

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a left hand trial is denoted 
    by 0, the start of a right hand trial by 1. Following the labelled trials above,
    the BCI was used for a cursor control task for *** minutes. The cursor 
    direction was self-selected, so these trials are labelled as -1.

    The duration of each trial is 1.5 seconds. There are 60 trials of each class.

    references
    ----------

    """


    def __init__(self):
        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2, unlabelled=3),
            code="OptGameMI",
            interval=[0, 1.5],
            paradigm="imagery",
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
        mi_dt = pd.read_csv(data_table_path / "optimization_data_table.csv")

        mi_dt = mi_dt[mi_dt["Task"] == "MI-G"]

        mi_dt.reset_index(inplace=True, drop=True)
        
        return Path("~/mne_data") / "MNE-opt-game-mi" / "{}.raw.fif". format(mi_dt["ParticipantLabel"][subject - 1])
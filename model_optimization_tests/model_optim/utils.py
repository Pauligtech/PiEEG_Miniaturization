import numpy as np
import pandas as pd
import mne

from sklearn.preprocessing import LabelEncoder, StandardScaler


def channels_to_channels_idx(channels, ch_names):
    assert (len(channels) > 0, "Please specify at least one channel")
    assert (len(ch_names) > 0, "Please specify a list of channel names to extract from")
    all_channels = ch_names
    channels_dict = {k: (True if k in channels else False) for k in all_channels}
    channels_idx = [i for i, v in enumerate(channels_dict.values()) if v]
    return channels_idx


def data_generator(dataset, subjects=[], channel_idx=[], filters=([8, 32],), sfreq=250):

    assert (dataset is not None, "Please specify a dataset")
    assert (len(subjects) > 0, "Please specify at least one subject")

    find_events = lambda raw, event_id: (
        mne.find_events(raw, shortest_event=0, verbose=False)
        if len(mne.utils._get_stim_channel(None, raw.info, raise_error=False)) > 0
        else mne.events_from_annotations(raw, event_id=event_id, verbose=False)[0]
    )

    data = dataset.get_data(subjects=subjects)

    X = []
    y = []
    metadata = []

    for subject_id in data.keys():
        for session_id in data[subject_id].keys():
            for run_id in data[subject_id][session_id].keys():
                raw = data[subject_id][session_id][run_id]

                for fmin, fmax in filters:
                    raw = raw.filter(
                        l_freq=fmin,
                        h_freq=fmax,
                        method="iir",
                        picks="eeg",
                        verbose=False,
                    )

                events = find_events(raw, dataset.event_id)

                tmin = dataset.interval[0]
                tmax = dataset.interval[1]

                channels = (
                    np.asarray(raw.info["ch_names"])[channel_idx]
                    if len(channel_idx) > 0
                    else np.asarray(raw.info["ch_names"])
                )

                stim_channels = mne.utils._get_stim_channel(
                    None, raw.info, raise_error=False
                )
                picks = mne.pick_channels(
                    raw.info["ch_names"],
                    include=channels,
                    exclude=stim_channels,
                    ordered=True,
                )

                x = mne.Epochs(
                    raw,
                    events,
                    event_id=dataset.event_id,
                    tmin=tmin,
                    tmax=tmax,
                    proj=False,
                    baseline=None,
                    preload=True,
                    verbose=False,
                    picks=picks,
                    event_repeated="drop",
                    on_missing="ignore",
                )
                x_events = x.events
                inv_events = {k: v for v, k in dataset.event_id.items()}
                labels = [inv_events[e] for e in x_events[:, -1]]

                x_resampled = x.resample(sfreq)  # Resampler_Epoch
                x_resampled_data = x_resampled.get_data(
                    copy=False
                )  # Convert_Epoch_Array
                x_resampled_data_standard_scaler = np.asarray(
                    [
                        StandardScaler().fit_transform(x_resampled_data[i])
                        for i in np.arange(x_resampled_data.shape[0])
                    ]
                )  # Standard_Scaler_Epoch

                n = x_resampled_data_standard_scaler.shape[0]
                met = pd.DataFrame(index=range(n))
                met["subject"] = subject_id
                met["session"] = session_id
                met["run"] = run_id
                x.metadata = met.copy()

                X.append(x)
                y.append(labels)
                metadata.append(met)

    return (
        np.concatenate(X, axis=0),
        np.concatenate(y),
        pd.concat(metadata, ignore_index=True),
    )

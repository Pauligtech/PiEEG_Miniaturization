import argparse
import time
import glob
import mne
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from custom_datasets.fatigue_mi import FatigueMI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sparsity", help="The level of sparsity (e.g., 0.1, 0.5, 0.9); default = 0.5"
)
parser.add_argument(
    "--quant", help="The quantization string (e.g., int8, float16); default = int8"
)
args = parser.parse_args()

user_given_sparsity = float(args.sparsity) if hasattr(args, "sparsity") else 0.5
user_given_quant = args.quant if hasattr(args, "quant") else "int8"

SKLRNG = 42


def data_generator(
    dataset, subjects=[1], channel_idx=[], filters=([8, 32],), sfreq=250
):

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

                # rpprint(channels)

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

                # rpprint({
                #     "X": np.asarray(x.get_data(copy=False)).shape,
                #     "y": np.asarray(labels).shape,
                #     "channels selected": np.asarray(raw.info['ch_names'])[channel_idx]
                # })

                # x.plot(scalings="auto")
                # display(x.info)

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

                # x_resampled.plot(scalings="auto")
                # display(x_resampled.info)

                n = x_resampled_data_standard_scaler.shape[0]
                # n = x.get_data(copy=False).shape[0]
                met = pd.DataFrame(index=range(n))
                met["subject"] = subject_id
                met["session"] = session_id
                met["run"] = run_id
                x.metadata = met.copy()

                # X.append(x_resampled_data_standard_scaler)
                X.append(x)
                y.append(labels)
                metadata.append(met)

    return (
        np.concatenate(X, axis=0),
        np.concatenate(y),
        pd.concat(metadata, ignore_index=True),
    )


def get_test_acc(model, train_test_data):
    X_test, y_test = train_test_data["X_test"], train_test_data["y_test"]

    def evaluate_model(interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # signatures = interpreter.get_signature_list()
        # rprint(interpreter.get_input_details(), interpreter.get_output_details(), signatures)

        # Run predictions on every image in the "test" dataset.
        predictions = []
        exec_times = []
        for i, v in enumerate(X_test):
            v = v[np.newaxis, :, :, np.newaxis].astype(np.float32)
            # if i % 1000 == 0:
            #   rprint('Evaluated on {n} results so far.'.format(n=i))
            # # Pre-processing: add batch dimension and convert to float32 to match with
            # # the model's input data format.
            # v = np.expand_dims(v, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, v)

            # Run inference.
            start_time = time.time()
            interpreter.invoke()
            exec_time = time.time() - start_time
            exec_times.append(exec_time)

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            class_prediction = np.argmax(output()[0])  # 0 = left, 1 = right
            predictions.append(class_prediction)

        print("\n")
        # Compare prediction results with ground truth labels to calculate accuracy.
        predictions = np.asarray(predictions)
        accuracy = (predictions == y_test).mean()
        avg_exec_time = np.mean(exec_times)
        return accuracy, avg_exec_time

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    test_accuracy, avg_exec_time = evaluate_model(interpreter)

    return {"test_accuracy": test_accuracy, "avg_exec_time": avg_exec_time}


fat_dataset = FatigueMI()

sparsity_level = user_given_sparsity
quantization = user_given_quant

model_files = glob.glob(
    f"./results/**/**/**/int8/pruned_model_{sparsity_level}_sparsity_{quantization}_quant.tflite"
)
model_info_files = glob.glob(f"./results/**/**/model_info.npy")

model_info_and_files = list(zip(model_info_files, model_files))

for _model_info, _model in model_info_and_files:

    model_info = np.load(_model_info, allow_pickle=True).item()

    X, y, _ = data_generator(
        fat_dataset,
        subjects=[model_info["subject"]],
        channel_idx=model_info["channels_idx_selected"],
        sfreq=model_info["sfreq"],
    )
    y_encoded = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=SKLRNG,
        shuffle=True,
        stratify=y_encoded,
    )
    train_test_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Load tflite model from disk
    model = open(_model, "rb").read()

    print(get_test_acc(model, train_test_data))

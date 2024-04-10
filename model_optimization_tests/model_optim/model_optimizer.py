import os
import uuid
import shutil
import time
import random
import datetime
import glob
import pickle
import tqdm
import copy
import optuna
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from rich import print as rprint
from rich.pretty import pprint as rpprint
from tqdm import tqdm
from itertools import chain
from functools import partial

# JAX + Keras
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["TF_USE_LEGACY_KERAS"] = "0"
# import jax
# import jax.numpy as jnp
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
)
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K

# Sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from sklearn import preprocessing

# Local
from .utils import channels_to_channels_idx, data_generator
from .models import (
    eeg_net,
    deep_conv_net,
    shallow_conv_net,
    lstm_net,
    lstm_cnn_net,
    lstm_cnn_net_v2,
)
from .callbacks import GetBest


class ModelOptimizer:
    def __init__(
        self,
        dataset,
        model_name="",
        approach="single_subject",
        **kwargs,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.approach = approach

        self.all_channels = self.dataset.get_data(subjects=[1])[1]["0"]["0"].info[
            "ch_names"
        ][:-1]

        self.models = {
            "eeg_net": eeg_net,
            "deep_conv_net": deep_conv_net,
            "shallow_conv_net": shallow_conv_net,
            "lstm_net": lstm_net,
            "lstm_cnn_net": lstm_cnn_net,
            "lstm_cnn_net_v2": lstm_cnn_net_v2,
        }
        self.model = self.models[self.model_name] if self.model_name else None

        if self.approach == "single_subject":
            self.subject = kwargs.get("subject", None)
            self.subjects = kwargs.get("subjects", [self.subject])
            self.channels = kwargs.get("channels", [])
            self.channels_idx = kwargs.get(
                "channels_idx",
                (
                    channels_to_channels_idx(self.channels, self.all_channels)
                    if len(self.channels) > 0
                    else []
                ),
            )

        self.SKLRNG = 42
        # self.RNG = jax.random.PRNGKey(self.SKLRNG)

    def get_subjects(self):
        return self.dataset.subject_list

    def objective_fn(
        self,
        trial,
        subjects=[],
        channels=None,
        model_str="shallow_conv_net",
        sfreq=128,
        batch_size=128,
        max_epochs=5,
        study_id=uuid.uuid4().hex,
        **kwargs,
    ):
        _SFREQ_ = (
            sfreq if sfreq else trial.suggest_categorical("sfreq", [128, 256, 300])
        )
        _TRAIN_SIZE_ = 0.8
        _TEST_SIZE_ = 1 - _TRAIN_SIZE_
        _BATCH_SIZE = (
            batch_size
            if batch_size
            else trial.suggest_int("batch_size", 32, 256, step=32)
        )

        using_all_subjects = len(subjects) == 0
        subjects = subjects if len(subjects) > 0 else self.dataset.subject_list
        model_fn = self.models[model_str]

        all_channels = self.all_channels
        if channels != []:
            channels_dict = (
                {
                    k: trial.suggest_categorical(f"channels_{k}", [True, False])
                    for k in all_channels
                }
                if channels == None
                else {k: True for k in channels}
            )
            channels_idx = [i for i, v in enumerate(channels_dict.values()) if v]
            while len(channels_idx) == 0:
                channels_dict = {
                    k: trial.suggest_categorical(f"channels_{k}", [True, False])
                    for k in all_channels
                }
                channels_idx = [i for i, v in enumerate(channels_dict.values()) if v]
        else:
            channels_idx = []

        X, y, _ = data_generator(
            self.dataset, subjects=subjects, channel_idx=channels_idx, sfreq=_SFREQ_
        )

        _NUM_SAMPLES_ = X.shape[-1]
        _NUM_CHANNELS_ = X.shape[-2]

        y_encoded = LabelEncoder().fit_transform(y)
        _NUM_CLASSES_ = len(np.unique(y_encoded))
        if "lstm" in model_str:
            X_train, X_test, y_train, y_test = train_test_split(
                X.reshape(-1, _NUM_SAMPLES_, _NUM_CHANNELS_),
                y_encoded,
                test_size=_TEST_SIZE_,
                random_state=self.SKLRNG,
                stratify=y_encoded,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=_TEST_SIZE_,
                random_state=self.SKLRNG,
                shuffle=True,
                stratify=y_encoded,
            )
        # X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, _NUM_SAMPLES_, _NUM_CHANNELS_), y_encoded, test_size=_TEST_SIZE_, random_state=SKLRNG, stratify=y_encoded)

        # y_train = keras.utils.to_categorical(y_train)
        # y_test = keras.utils.to_categorical(y_test)

        model_params = {
            "nb_classes": _NUM_CLASSES_,
            "channels": _NUM_CHANNELS_,
            "samples": _NUM_SAMPLES_,
        }

        if model_str == "shallow_conv_net":
            model_params = {
                **model_params,
                "pool_size_d2": trial.suggest_int("pool_size_d2", 5, 95, step=5),
                "strides_d2": trial.suggest_int("strides_d2", 1, 31, step=1),
                "conv_filters_d2": trial.suggest_int("conv_filters_d2", 5, 55, step=1),
                "conv2d_1_units": trial.suggest_int("conv2d_1_units", 10, 200, step=10),
                "conv2d_2_units": trial.suggest_int("conv2d_2_units", 10, 200, step=10),
                "l2_reg_1": trial.suggest_float("l2_reg_1", 0.001, 0.9),
                "l2_reg_2": trial.suggest_float("l2_reg_2", 0.001, 0.9),
                "l2_reg_3": trial.suggest_float("l2_reg_3", 0.001, 0.9),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.9, step=0.1),
            }

        if model_str == "eeg_net":
            model_params = {
                **model_params,
                "conv2d_1_units": trial.suggest_int("conv2d_1_units", 8, 64, step=8),
                "conv2d_1_kernl_length": trial.suggest_int(
                    "conv2d_1_kernl_length", 16, 128, step=16
                ),
                "pool_1_size": trial.suggest_int("pool_1_size", 4, 16, step=4),
                "conv2d_depth_multiplier": trial.suggest_int(
                    "conv2d_depth_multiplier", 2, 8, step=2
                ),
                "conv2d_2_units": trial.suggest_int("conv2d_2_units", 8, 64, step=8),
                "conv2d_2_kernl_length": trial.suggest_int(
                    "conv2d_2_kernl_length", 16, 128, step=16
                ),
                "pool_2_size": trial.suggest_int("pool_2_size", 4, 16, step=4),
                "l2_reg_1": trial.suggest_float("l2_reg_1", 0.001, 0.9),
                "l2_reg_2": trial.suggest_float("l2_reg_2", 0.001, 0.9),
                "l2_reg_3": trial.suggest_float("l2_reg_3", 0.001, 0.9),
                "l2_reg_4": trial.suggest_float("l2_reg_4", 0.001, 0.9),
                "dropout_rate_1": trial.suggest_float(
                    "dropout_rate", 0.1, 0.9, step=0.1
                ),
                "dropout_rate_2": trial.suggest_float(
                    "dropout_rate", 0.1, 0.9, step=0.1
                ),
            }

        if model_str == "lstm_cnn_net":
            model_params = {
                **model_params,
                "conv1d_1_units": trial.suggest_int("conv1d_1_units", 10, 200, step=10),
                "conv1d_1_kernel_size": trial.suggest_int(
                    "conv1d_1_kernel_size", 1, 100, step=1
                ),
                "conv1d_1_strides": trial.suggest_int(
                    "conv1d_1_strides", 1, 50, step=1
                ),
                # "conv1d_1_maxpool_size": trial.suggest_int("conv1d_1_maxpool_size", 1, 100, step=2),
                "conv1d_1_maxpool_strides": trial.suggest_int(
                    "conv1d_1_maxpool_strides", 1, 4, step=1
                ),
                "lstm_1_units": trial.suggest_int("lstm_1_units", 10, 200, step=10),
                "l2_reg_1": trial.suggest_float("l2_reg_1", 0.001, 0.8),
                "l2_reg_2": trial.suggest_float("l2_reg_2", 0.001, 0.8),
                "l2_reg_3": trial.suggest_float("l2_reg_3", 0.001, 0.8),
                "dropout_rate_1": trial.suggest_float(
                    "dropout_rate_1", 0.1, 0.9, step=0.1
                ),
                "dropout_rate_2": trial.suggest_float(
                    "dropout_rate_2", 0.1, 0.9, step=0.1
                ),
            }

        model = model_fn(**model_params)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="rmsprop",
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        training_start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            batch_size=_BATCH_SIZE,
            epochs=max_epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=[
                GetBest(monitor="val_accuracy", verbose=1, mode="auto"),
                # keras.callbacks.EarlyStopping(
                #     monitor="val_loss", patience=75, restore_best_weights=True
                # ),
                # keras.callbacks.ReduceLROnPlateau(
                #     monitor="val_loss", patience=75, factor=0.5
                # ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", patience=3, factor=0.1
                ),
            ],
        )
        training_end_time = time.time()

        print("\n")

        # Rename history.history["sparse_categorical_accuracy"] to history.history["accuracy"]
        # history.history["accuracy"] = history.history["sparse_categorical_accuracy"]
        # history.history["val_accuracy"] = history.history["val_sparse_categorical_accuracy"]

        # Save X_test, y_test to ./temp/{subject}/{study_id}/data
        # test_data = {
        #     "X_test": X_test,
        #     "y_test": y_test,
        # }
        # np.save(
        #     f"./temp/{subjects if not using_all_subjects else '[]'}/{study_id}/data/test_data_{trial.number}.npy",
        #     test_data,
        #     allow_pickle=True,
        # )

        inference_start_time = time.time()
        test_eval = model.evaluate(X_test, y_test, batch_size=_BATCH_SIZE)
        inference_end_time = time.time()

        trial.set_user_attr(
            "trial_data",
            {
                "subjects": subjects,
                "sklearn_rng": self.SKLRNG,
                "channels_selected": np.asarray(self.all_channels)[channels_idx],
                "channels_selected_idx": channels_idx,
                "history": history.history,
                "model": model.to_json(),
                "weights": model.get_weights(),
                "train_accuracy": history.history["accuracy"],
                "val_accuracy": history.history["val_accuracy"],
                "test_accuracy": test_eval[1],
                "train_loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "test_loss": test_eval[0],
                "data_path": f"./temp/{subjects}/{study_id}/data/",
                "batch_size": batch_size,
                "num_training_epochs": max_epochs,
                "model_name": model_str,
                "sfreq": sfreq,
                "training_time": training_end_time - training_start_time,
                "inference_time": inference_end_time - inference_start_time,
            },
        )

        print("\n")
        val_acc_nd_array = np.asarray(
            history.history["val_accuracy"]
            # history.history["val_accuracy"][-(max_epochs // 2) :]
        )
        max_val_acc = val_acc_nd_array.max()
        train_acc_for_max_val_acc = history.history["accuracy"][
            val_acc_nd_array.argmax()
        ]
        # # max_train_acc = np.asarray(history.history["accuracy"]).max()

        # # cost = np.min(history.history["val_loss"]) + (1 - max_val_acc) ** 2
        # # cost = np.abs(1 - max_val_acc)
        # cost = np.min(history.history["val_loss"][-5:]) + np.abs(1 - max_val_acc)
        # # cost = np.min(history.history["val_loss"][-(max_epochs // 2) :])
        # # cost = np.min(history.history["val_loss"][-20:])  # + (1 - max_val_acc) ** 2
        # # cost = np.abs(1 - max_val_acc)
        # # cost = (1 - max_val_acc) ** 2
        # if max_val_acc > train_acc_for_max_val_acc:
        #     cost += 0.25

        # L1 = 1e-3 * (len(channels_idx) / len(all_channels))
        # # L1 = 5e-5 * (len(channels_idx))
        # cost += L1

        # if not (0.75 <= train_acc_for_max_val_acc <= 1.00):
        #     cost += 0.75

        cost = (1 - max_val_acc) ** 2
        L1 = 1e-3 * (len(channels_idx) / len(all_channels))
        cost += L1

        if max_val_acc > train_acc_for_max_val_acc:
            cost += 0.25
        if not (0.75 <= train_acc_for_max_val_acc <= 0.99):
            cost += 0.75

        return cost

    def early_stopping_check(self, study, trial, early_stopping_rounds=10):
        current_trial_number = trial.number
        best_trial_number = study.best_trial.number
        should_stop = (
            current_trial_number - best_trial_number
        ) >= early_stopping_rounds
        if should_stop:
            optuna.logging._get_library_root_logger().info(
                "Early stopping detected: %s", should_stop
            )
            study.stop()

    def heuristic_optimizer(
        self,
        obj_fn,
        max_iter=25,
        max_epochs=5,
        show_progress_bar=True,
        subjects=[],
        channels=None,
        model_str="shallow_conv_net",
        sfreq=128,
        batch_size=128,
        max_stag_count=10,
        study_id=uuid.uuid4().hex,
        enqueue_prev_best_trial=False,
        **kwargs,
    ):

        obj_fn = obj_fn or self.objective_fn
        subjects = subjects if subjects != None else self.subjects
        model_str = model_str if model_str != None else self.model_str
        # sfreq = sfreq if sfreq != None else self.sfreq
        # batch_size = batch_size if batch_size != None else self.batch_size

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        objective = partial(
            obj_fn,
            subjects=subjects,
            channels=channels,
            model_str=model_str,
            sfreq=sfreq,
            batch_size=batch_size,
            study_id=study_id,
            max_epochs=max_epochs,
        )

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.NSGAIISampler()
        )
        if enqueue_prev_best_trial:
            study.enqueue_trial(
                self.latest_study.best_trial.params,
                user_attrs={
                    "trial_data": self.latest_study.best_trial.user_attrs["trial_data"]
                },
            )

        try:
            study.optimize(
                objective,
                n_trials=max_iter,
                show_progress_bar=show_progress_bar,
                callbacks=[
                    partial(
                        self.early_stopping_check,
                        early_stopping_rounds=max_stag_count,
                    )
                ],
                **kwargs,
            )
        except KeyboardInterrupt:
            pass

        return study

    def search_best_model(
        self,
        model_name=None,
        rounds=1,
        subjects=[],
        channels=None,
        sfreq=None,
        batch_size=None,
        max_iter=50,
        max_stag_count=10,
        max_epochs=5,
        replace_previous_study_for_subjects=True,
        save_best_trial_only=False,
    ):

        study_id = uuid.uuid4().hex
        model_str = model_name if model_name != None else self.model_name

        if replace_previous_study_for_subjects:
            # Remove ./temp/{subjects} if it exists
            if os.path.exists(f"./temp/{subjects}"):
                shutil.rmtree(f"./temp/{subjects}")

        # Make temp/{study_id}/[data,model,config] if it does not exist
        for path in [
            f"./temp/{subjects}/{study_id}/data",
            f"./temp/{subjects}/{study_id}/model",
        ]:
            if not os.path.exists(path):
                os.makedirs(path)

        for round in range(rounds):

            study = self.heuristic_optimizer(
                self.objective_fn,
                max_iter=max_iter,
                subjects=subjects,
                channels=channels,
                model_str=model_str,
                sfreq=sfreq,
                batch_size=batch_size,
                max_stag_count=max_stag_count,
                max_epochs=max_epochs,
                study_id=study_id,
                enqueue_prev_best_trial=False if round == 0 else True,
            )

            if not save_best_trial_only:
                np.save(
                    f"./temp/{subjects}/{study_id}/model/{model_str}_study.npy",
                    study,
                    allow_pickle=True,
                )
            np.save(
                f"./temp/{subjects}/{study_id}/model/{model_str}_study_best_trial.npy",
                study.best_trial,
                allow_pickle=True,
            )

            # Clean up
            # Delete all ./temp/**/** that contain no files
            # for root, dirs, files in os.walk("./temp"):
            #     if not files and not dirs:
            #         shutil.rmtree(root)

            self.latest_study = study

        return study

    def get_study_metrics(self, study):
        study = study if study != None else self.latest_study

        trial_metrics_dict = {
            "train_acc": [],
            "test_acc": [],
            "val_acc": [],
            "train_val_acc_diff": [],
            "train_loss": [],
            "val_loss": [],
            "train_val_loss_diff": [],
            "test_loss": [],
            "scores": [],
            "channels_selected": [],
            "sfreq": [],
            "batch_size": [],
        }
        for i, trial in enumerate(study.trials_dataframe().itertuples()):
            trial_user_attrs = trial.user_attrs_trial_data
            trial_metrics_dict["scores"].append(trial.value)
            trial_metrics_dict["train_acc"].append(
                np.max(trial_user_attrs["train_accuracy"])
            )
            trial_metrics_dict["test_acc"].append(trial_user_attrs["test_accuracy"])
            trial_metrics_dict["val_acc"].append(
                np.max(trial_user_attrs["val_accuracy"])
            )
            trial_metrics_dict["channels_selected"].append(
                trial_user_attrs["channels_selected"]
            )
            trial_metrics_dict["train_val_acc_diff"].append(
                abs(
                    np.max(trial_user_attrs["train_accuracy"])
                    - np.max(trial_user_attrs["val_accuracy"])
                )
            )
            (
                trial_metrics_dict["train_loss"].append(
                    np.min(trial_user_attrs["train_loss"])
                    if "train_loss" in trial_user_attrs
                    else None
                )
            )
            (
                trial_metrics_dict["val_loss"].append(
                    np.min(trial_user_attrs["val_loss"])
                    if "val_loss" in trial_user_attrs
                    else None
                )
            )
            (
                trial_metrics_dict["train_val_loss_diff"].append(
                    abs(
                        np.min(trial_user_attrs["train_loss"])
                        - np.min(trial_user_attrs["val_loss"])
                    )
                    if "train_loss" in trial_user_attrs
                    and "val_loss" in trial_user_attrs
                    else None
                )
            )
            (
                trial_metrics_dict["test_loss"].append(
                    np.min(trial_user_attrs["test_loss"])
                    if "test_loss" in trial_user_attrs
                    else None
                )
            )
            trial_metrics_dict["sfreq"].append(
                trial.params_sfreq if hasattr(trial, "params_sfreq") else None
            )
            trial_metrics_dict["batch_size"].append(
                trial.params_batch_size if hasattr(trial, "params_batch_size") else None
            )
        trial_metrics_df = pd.DataFrame(trial_metrics_dict)

        return trial_metrics_df

    def clean_up(
        self,
        subjects=None,
        best_study_id=None,
        best_trial_id=None,
        action="remove_all_but_best_trial_data",
    ):
        subjects = subjects if subjects != None else self.subjects
        subjects_glob_str = "[[]" + str(subjects).strip("[]") + "[]]"
        if action == "remove_all_but_best_trial_data":
            if best_study_id == None:
                subjects_best_trials = glob.glob(
                    f"./temp/{subjects_glob_str}/*/model/study_best_trial.npy"
                )
                rpprint(subjects_best_trials)
                if len(subjects_best_trials) > 0:
                    rprint(
                        f"Found {len(subjects_best_trials)} best trials for subjects: {subjects}"
                    )
                best_study_id = subjects_best_trials[0].split("/")[-3]
            if best_trial_id == None:
                best_trial = np.load(
                    f"./temp/{subjects}/{best_study_id}/model/study_best_trial.npy",
                    allow_pickle=True,
                )
                best_trial_id = best_trial.item().number
            else:
                best_trial = (
                    np.load(
                        f"./temp/{subjects}/{best_study_id}/model/study_trials.npy",
                        allow_pickle=True,
                    )
                    .item()
                    .trials[best_trial_id]
                )
                best_trial_id = best_trial.number
            # Remove all test_data_[trial_id].npy files except for the best_trial_id
            for file in glob.glob(
                f"./temp/{subjects_glob_str}/{best_study_id}/data/test_data_*.npy"
            ):
                if int(file.split("_")[-1].split(".")[0]) != best_trial_id:
                    os.remove(file)

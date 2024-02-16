import os
import pandas as pd
from pathlib import Path


def GenerateDataTable(dataFolderPath):
    # dataFolderPath = Path("../"+dataFolderPath)

    data_columns = ["ParticipantLabel", "SessionNumber", "Task", "Filename"]
    data_table = pd.DataFrame(columns=data_columns)

    P_list = os.listdir(dataFolderPath)

    for p in P_list:
        # get the parcticipant ID
        p_name = p.split("_")[0]

        sub_p_names = os.listdir(dataFolderPath / p)
        for sub_p_name in sub_p_names:
            try:
                S_list = os.listdir(dataFolderPath / p / sub_p_name)
                for s in S_list:
                    s_number = int(s.split("S")[-1])
                    try:
                        T_list = os.listdir(dataFolderPath / p / sub_p_name / s / "eeg")
                        for t in T_list:
                            this_path = str(dataFolderPath / p / sub_p_name / s / "eeg" / t)

                            if "_old" in t:
                                continue

                            if "SMR-S" in t:
                                temp_df = pd.DataFrame(
                                    [[p_name, s_number, "MI-S", this_path]],
                                    columns=data_columns,
                                    index=None,
                                )
                                data_table = pd.concat([data_table, temp_df])

                            if "SMR-G" in t:
                                temp_df = pd.DataFrame(
                                    [[p_name, s_number, "MI-G", this_path]],
                                    columns=data_columns,
                                    index=None,
                                )
                                data_table = pd.concat([data_table, temp_df])

                            if "P300-S" in t:
                                temp_df = pd.DataFrame(
                                    [[p_name, s_number, "P300-S", this_path]],
                                    columns=data_columns,
                                    index=None,
                                )
                                data_table = pd.concat([data_table, temp_df])

                            if "P300-G" in t:
                                temp_df = pd.DataFrame(
                                    [[p_name, s_number, "P300-G", this_path]],
                                    columns=data_columns,
                                    index=None,
                                )
                                data_table = pd.concat([data_table, temp_df])

                    except:
                        print("")
            except:
                print("")

    data_table.reset_index(inplace=True, drop=True)
    return data_table

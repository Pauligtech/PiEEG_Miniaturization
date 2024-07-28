import os
import pandas as pd
from pathlib import Path

def GenerateDataTable(dataFolderPath):
    # dataFolderPath = Path("../"+dataFolderPath)

    data_columns = [
        "ParticipantLabel",
        "ParticipantNumber" "SessionNumber",
        "Task",
        "Filename",
    ]
    data_table = pd.DataFrame(columns=data_columns, index=None)

    P_list = os.listdir(dataFolderPath)

    for p in P_list:
        # get the parcticipant ID
        p_name = p.split("_")[0]

        try:
            T_list = os.listdir(dataFolderPath / p)
            for t in T_list:
                try:
                    this_xdf = os.listdir(dataFolderPath / p / t / "eeg")[0]
                    this_path = dataFolderPath / p / t / "eeg" / this_xdf
                except:
                    continue

                if "ses-MI" in t:
                    temp_df = pd.DataFrame(
                        [[p_name, 1, "MI-S", this_path]],
                        columns=data_columns,
                        index=None,
                    )
                    data_table = pd.concat([data_table, temp_df])

                # if "ses-preRS_mi" in t:
                #     PT_df.loc[p_name,"MI_RS_Pre"] = this_path

                # if "ses-postRS_mi" in t:
                #     PT_df.loc[p_name,"MI_RS_Post"] = this_path

                # if "ses-preBB_mi" in t:
                #     PT_df.loc[p_name,"MI_BB_Pre"] = this_path

                # if "ses-postBB_mi" in t:
                #     PT_df.loc[p_name,"MI_BB_Post"] = this_path

                if "ses-P300" in t:
                    temp_df = pd.DataFrame(
                        [[p_name, 1, "P300-S", this_path]],
                        columns=data_columns,
                        index=None,
                    )
                    data_table = pd.concat([data_table, temp_df])

                # if "ses-preRS_p300" in t:
                #     PT_df.loc[p_name,"P300_RS_Pre"] = this_path

                # if "ses-postRS_p300" in t:
                #     PT_df.loc[p_name,"P300_RS_Post"] = this_path

                # if "ses-preBB_p300" in t:
                #     PT_df.loc[p_name,"P300_BB_Pre"] = this_path

                # if "ses-postBB_p300" in t:
                #     PT_df.loc[p_name,"P300_BB_Post"] = this_path

                # if "ses-VIDEO" in t:
                #     PT_df.loc[p_name,"VIDEO"] = this_path

                # if "ses-preRS_vid" in t:
                #     PT_df.loc[p_name,"Video_RS_Pre"] = this_path

                # if "ses-postRS_vid" in t:
                #     PT_df.loc[p_name,"Video_RS_Post"] = this_path

                # if "ses-preBB_vid" in t:
                #     PT_df.loc[p_name,"Video_BB_Pre"] = this_path

                # if "ses-postBB_vid" in t:
                #     PT_df.loc[p_name,"Video_BB_Post"] = this_path
        except:
            print("")

    data_table.reset_index(inplace=True, drop=True)
    return data_table


def GenerateDF(dataFolder):
    dataFolderPath = Path(dataFolder)

    # List the different subjects
    P_list = os.listdir(dataFolderPath)

    # Create a DataFrame with all of the participants and all of the tasks
    tasks = [
        "MI",
        "MI_RS_Pre",
        "MI_RS_Post",
        "MI_BB_Pre",
        "MI_BB_Post",
        "P300",
        "P300_RS_Pre",
        "P300_RS_Post",
        "P300_BB_Pre",
        "P300_BB_Post",
        "VIDEO",
        "Video_RS_Pre",
        "Video_RS_Post",
        "Video_BB_Pre",
        "Video_BB_Post",
    ]
    PT_df = pd.DataFrame(columns=tasks)

    print(P_list)
    print("Loading data from {} participants".format(len(P_list)))

    # List of tasks

    for p in P_list:
        # get the parcticipant ID
        p_name = p.split("_")[0]

        T_list = os.listdir(dataFolderPath / p)
        for t in T_list:
            try:
                this_xdf = os.listdir(dataFolderPath / p / t / "eeg")[0]
                this_path = dataFolderPath / p / t / "eeg" / this_xdf
            except:
                continue

            if "ses-MI" in t:
                PT_df.loc[p_name, "MI"] = this_path

            if "ses-preRS_mi" in t:
                PT_df.loc[p_name, "MI_RS_Pre"] = this_path

            if "ses-postRS_mi" in t:
                PT_df.loc[p_name, "MI_RS_Post"] = this_path

            if "ses-preBB_mi" in t:
                PT_df.loc[p_name, "MI_BB_Pre"] = this_path

            if "ses-postBB_mi" in t:
                PT_df.loc[p_name, "MI_BB_Post"] = this_path

            if "ses-P300" in t:
                PT_df.loc[p_name, "P300"] = this_path

            if "ses-preRS_p300" in t:
                PT_df.loc[p_name, "P300_RS_Pre"] = this_path

            if "ses-postRS_p300" in t:
                PT_df.loc[p_name, "P300_RS_Post"] = this_path

            if "ses-preBB_p300" in t:
                PT_df.loc[p_name, "P300_BB_Pre"] = this_path

            if "ses-postBB_p300" in t:
                PT_df.loc[p_name, "P300_BB_Post"] = this_path

            if "ses-VIDEO" in t:
                PT_df.loc[p_name, "VIDEO"] = this_path

            if "ses-preRS_vid" in t:
                PT_df.loc[p_name, "Video_RS_Pre"] = this_path

            if "ses-postRS_vid" in t:
                PT_df.loc[p_name, "Video_RS_Post"] = this_path

            if "ses-preBB_vid" in t:
                PT_df.loc[p_name, "Video_BB_Pre"] = this_path

            if "ses-postBB_vid" in t:
                PT_df.loc[p_name, "Video_BB_Post"] = this_path

    PT_df.head()
    return PT_df

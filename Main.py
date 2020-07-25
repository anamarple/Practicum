import pandas as pd
import numpy as np
import os
import shutil
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import time
from Path_Reader import read
from Normalize import normalize_df
from Compare import predict_b1, predict_b2, get_cv_l1, get_cv_l2
from Predict import get_folder


# Takes in source location from user, creates df from path and file names, normalizes df, and feeds into model
# Input: string of source location where unorganized data resides
# Returns: df
def classify(source):
    df = read(source)
    norm_df = normalize_df(df)

    # Extract features for each bucket level
    # Level 1:
    cv_l1 = get_cv_l1()
    features_l1 = cv_l1.transform(np.array(norm_df['Normalized_Path']))

    # Level 2:
    [cv_l2_comp, cv_l2_geo, cv_l2_prod, cv_l2_gen] = get_cv_l2()
    features_l2_comp = cv_l1.transform(np.array(norm_df['Normalized_File']))
    features_l2_geo = cv_l2_geo.transform(np.array(norm_df['Normalized_File']))
    features_l2_prod = cv_l2_prod.transform(np.array(norm_df['Normalized_File']))
    features_l2_gen = cv_l2_gen.transform(np.array(norm_df['Normalized_File']))

    # Predict 1st bucket
    y = predict_b1(features_l1)
    norm_df['Bucket1'] = y

    # Predict 2nd bucket
    z = predict_b2(norm_df, 'Bucket1', cv_l2_comp, cv_l2_geo, cv_l2_prod, cv_l2_gen, )
    norm_df['Bucket2'] = z

    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)

    return norm_df


# Copies files and places in destination folder under appropriate sub folder
# Input: df (from previous method) and string of destination location of where to copy classified files
# Returns: nothing but a message to user that program is done
def place_in_dest(df, destination):

    # Set up window and progress bar
    master = tk.Tk()
    progress_bar = ttk.Progressbar(master, orient = 'horizontal', mode = 'determinate', maximum = len(df), value = 0)
    label = tk.Label(master, text = 'Copying files...')
    label.pack()
    progress_bar.pack()
    master.update()

    files = ['00 Previous', '01 General', '02 Volumetric and Reserves Estimates', '03 Production',
             '04 Development Plans',
             '05 Economics', '06 Field Reports', '07 Seismic Data', '08 Geologic Maps', '09 Bubble Maps',
             '10 PVT and Text Data',
             '11 Petrophysical Summaries', '12 Cross-Sections', '13 Logs', '15 Field Activity', '16 Modeling']

    # Create sub folders in destination if they don't exist already
    for f in files:
        try:
            os.mkdir(destination + '\\' + f)
        # Folder already exists
        except FileExistsError:
            next

    # Master draws progress bar
    master.update()
    progress_bar['value'] = 0
    master.update()

    while progress_bar['value'] < len(df):
        for idx in df.index:
            source = str(df['Path'][idx]) + '\\' + str(df['File'][idx])  # source_path\filename.ext
            folder = get_folder(str(df['Bucket2'][idx]))
            target = destination + '\\' + folder + '\\' + str(df['File'][idx])  # dest_path\folder\filename.ext

            # This might take a minute as files are being copied over
            # Update progress bar
            progress_bar['value'] += 1
            master.update()
            time.sleep(0.5)

            shutil.copyfile(source, target)

    # Message to user that files have been copied over
    message = messagebox.showinfo(title = None, message = "Done!")

    tk.mainloop(1)
    return

import subprocess
import sys
import pyodbc
from Main import classify, place_in_dest
import pandas as pd
import numpy as np

execLoc = r'C:\Program Files\IronPython 2.7\ipy.exe'
scriptLoc = r'C:/Users/amarple/PycharmProjects/Practicum/IronPython_Interact.py'


# Adds df to SQL Express database to store categorizations to build/improve future model
def add_df(df):
    # Connect to database
    conn_str = r"Driver={SQL Server};Server=dm-sqlexpress\sqlexpress;Database=DMFileClassification" \
               r";UID=DMFCUser;PWD=Ydo%A39&B0Sl; "
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Remove duplicates first
    for index, row in df.iterrows():
        with conn.cursor() as crsr:
            row = crsr.execute("DELETE FROM [dbo].[Data] "
                               "WHERE ([Path] = '" + row["Path"] + "') "
                                                                   "AND ([File] = '" + row["File"] + "')")

    # Insert df into 'Data' table in SQL Express DB
    cursor.executemany(f"INSERT INTO [dbo].[Data] ([Path], [File], [Bucket], [Bucket2]) VALUES (?, ?, ?, ?)",
                       df.itertuples(index=False))

    conn.commit()
    conn.close()
    return


# Source: https://www.digitalocean.com/community/tutorials/how-to-use-subprocess-to-run-external-programs-in-python-
# 3#capturing-output-from-an-external-program
try:
    # Show form to user, get source and destination paths
    output = subprocess.run([execLoc, scriptLoc], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=False, creationflags=0x08000000)
    output = output.stdout.decode('utf-8').split("\r\n")

    sourcePath = output[0]
    destPath = output[1]

    # Call classify function
    df = classify(sourcePath)
    df = df.drop(['Normalized_Path', 'Normalized_File'], axis=1)

    # Convert df to bytes string
    df_list = df.values.tolist()
    df_byte = []
    for i in df_list:
        for j in i:
            enc = j.encode('utf-8')
            df_byte.append(enc)
            df_byte.append(b',')
        df_byte.append(b'|')
    result = b"*".join(df_byte)

    # Return df to user, give them chance to fix categorizations
    resultsLoc = r'C:/Users/amarple/PycharmProjects/Practicum/IronPython_Results.py'
    x = subprocess.Popen([execLoc, resultsLoc], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=False, creationflags=0x08000000)
    x.stdin.write(result)
    output = x.communicate()[0]
    x.stdin.close()

    # Read output from IronPython_Results script into df
    output = output.decode('utf-8').split("\r\n")
    output.remove("")
    output_array = np.array(output)
    output_reshape = np.reshape(output_array, (-1, 4))
    df_final = pd.DataFrame(output_reshape, columns=['Path', 'File', 'Bucket', 'Bucket2'])

    # Add df to database
    add_df(df_final)

    # Copy files over to destination
    place_in_dest(df_final, destPath)


# User exits out of wpf form
except IndexError:
    sys.exit()

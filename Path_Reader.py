import os
import pandas as pd


# Looks through each file in root provided and adds path and file name to df
# Input: String of root file location
# Returns: data frame
def read(root):

    path = []
    filename = []
    for dir, subdir, files in os.walk(root):

        # If files is not empty
        if files:
            for f in files:
                path.append(dir)
                filename.append(f)

    data = {'Path': path, 'File': filename}
    df = pd.DataFrame(data)

    return df
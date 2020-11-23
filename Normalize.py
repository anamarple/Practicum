import re
import string


# Pre-processes / normalizes file path and name
# Input: data frame
# Returns: data frame with normalized path column added
def normalize_df(df):
    # Pre-process path name: Delete everything in path before and at _DM_DATA\ OR 00 Incoming Data\
    # OR 00 Incoming Data\mm-dd-yyyy\ OR 01 Data from Client\ OR FTP_Root
    path_norm_list = []
    for i in df['Path']:

        # Look for '_DM_DATA'
        index = i.find('_DM_DATA')
        if index == -1:
            # Look for 'MM-DD-YYYY' format
            index = re.search(r'(\d{2})-(\d{2})-(\d{4})', i)
            if not index:
                # Look for '00 Incoming Data'
                index = i.find('00 Incoming Data')
                if index == -1:
                    # Look for '01 Data from Client'
                    index = i.find('01 Data from Client')
                    if index == -1:
                        # Look for 'FTP_Root'
                        index = i.find('FTP_Root')
                        if index == -1:
                            # print('Nothing found ', i)
                            path_norm_list.append('')

                        else:
                            path_norm = i[index + 9::]
                            path_norm_list.append(path_norm)
                    else:
                        path_norm = i[index + 20::]
                        path_norm_list.append(path_norm)
                else:
                    path_norm = i[index + 17::]
                    path_norm_list.append(path_norm)
            else:
                path_norm = i[index.start() + 11::]
                path_norm_list.append(path_norm)
        else:
            path_norm = i[index + 9::]
            path_norm_list.append(path_norm)

    df['Normalized_Path'] = path_norm_list
    '''
    for i in df.index:
        print('Path: ', df['Path'][i])
        print('Reduced Path: ', df['Normalized_Path'][i])
    '''
    return df


# Removes punctuation from normalized path and replaces, removes punctuation from file name and adds as
# 'normalized file' column
# Input: data frame (assumes it already includes 'normalized path' column)
# Returns: data frame (with punctuation removed from 'normalized path' column and new 'normalized file' column)
def remove_punc(df):

    # Pre-process path name: removes punctuation and replaces
    path_norm = []
    for j in df['Normalized_Path']:
        norm = j.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        path_norm.append(norm)
    df['Normalized_Path'] = path_norm

    # Pre-process file name: removes punctuation and makes new column
    file_norm = []
    for j in df['File']:
        norm = j.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        file_norm.append(norm)
    df['Normalized_File'] = file_norm

    # Add normalized file name to end of normalized path name
    for idx in df.index:
        # if str(df['Normalized_Path'][idx]) == '':
        df['Normalized_Path'][idx] = df['Normalized_Path'][idx] + ' ' + df['Normalized_File'][idx]

    return df

import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
from collections import Counter

'''
conn_str = r"Driver={SQL Server};Server=dm-sqlexpress\sqlexpress;Database=DMFileClassification;" \
           "UID=DMFCUser;PWD=Ydo%A39&B0Sl;"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

SQL_Query = pd.read_sql_query("SELECT * FROM [dbo].[Data]", conn)
df = pd.DataFrame(SQL_Query, columns=['File', 'Path', 'Bucket', 'Bucket2'])


# Generate stats of file extensions
ext_list = []
for i in df['File']:
    # Look for LAST '.'
    index = i.rfind('.')
    ext = i[index + 1::]
    ext_list.append(ext)

# for j in ext_list:
#    print(j)


df_ = pd.DataFrame({'freq': ext_list})
df_.groupby('freq', as_index=False).size().plot(kind='bar')
# plt.show()

c = Counter(ext_list)
# print(c)
# Top: *xml, *pdf, *xlsm, *bxml, zat, dat, xlsx, bin, ptd, pptx, las, Bulk, dlis...
'''

# TODO: If confidence of prediction is below a certain threshold, read contents of file depending on extension?

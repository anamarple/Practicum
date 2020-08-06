import sys
import tkinter as tk
from tkinter import messagebox
import pyodbc
from pandastable import Table
from pandastable.dialogs import addButton
from Main import classify, place_in_dest
import os


# Shows window to user w/ text entries for 'source' (where files are located) and 'destination' (where categorized
# files will be copied over)
class input_locations:

    def submit(self):
        self.source = self.e1.get()
        self.destination = self.e2.get()

        self.master.destroy()
        return

    def cancel(self):
        self.master.destroy()
        sys.exit()

    def __init__(self):
        self.master = tk.Tk()
        self.master.geometry('200x75')

        # Add labels
        tk.Label(self.master, text = 'Source').grid(row = 0)
        tk.Label(self.master, text = 'Destination').grid(row = 2)

        # Add entries
        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        self.e1 = tk.Entry(self.master, textvariable = self.source)
        self.e2 = tk.Entry(self.master, textvariable = self.destination)
        self.e1.grid(row = 0, column = 1)
        self.e2.grid(row = 2, column = 1)

        # Add 'OK' button
        self.b1 = tk.Button(self.master, text = 'OK', command = self.submit)
        self.b1.grid(row = 3, column = 1)

        # Direct to cancel method if 'X' button pressed
        self.master.protocol("WM_DELETE_WINDOW", self.cancel)

        self.master.mainloop(1)


# Lets user view pandas table of results and make corrections (if necessary) before calling 'place_in_dest' method to
# copy over files
class results:

    def ok_button(self):
        # update df if any buckets were changed
        self.df_new = self.pt.model.df
        self.root.destroy()

    def __init__(self, df):
        self.root = tk.Tk()
        self.root.title('Results')

        # Add canvas & table
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill = 'both', expand = True)
        self.pt = Table(self.canvas)
        self.pt.show()
        self.pt.model.df = df

        # Add toolbar
        self.toolbar = tk.Canvas(self.root)
        self.toolbar.pack()
        addButton(self.toolbar, 'OK', self.ok_button, side = 'bottom')

    def popup(self, event):
        menu = self.pt.popupMenu(event = event)

    def run(self):
        self.pt.bind("<Button-3>", self.popup)
        self.root.mainloop(1)


# Adds df to SQL Express database to store categorizations to build/improve future model
def add_df(df):
    # Connect to database
    conn_str = r"Driver={SQL Server};Server=dm-sqlexpress\sqlexpress;Database=DMFileClassification" \
               r";UID=DMFCUser;PWD=Ydo%A39&B0Sl; "
    # conn_str = r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=V:\APLA\users\arm\File " \
    #           r"Classifications DB.accdb "
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
                       df.itertuples(index = False))

    conn.commit()
    conn.close()
    return


# Driver for user interface
def run():
    # Retrieve source and destination locations entered by user
    try:
        i = input_locations()
        source = i.source
        destination = i.destination

    except AttributeError or TypeError:
        exit()

    try:
        # Source and destination are valid directories
        if os.path.isdir(source) and os.path.isdir(destination):

            # Call classify function
            df = classify(source)
            df = df.drop(['Normalized_Path', 'Normalized_File'], axis = 1)

            # Copy files over to destination
            r = results(df)
            r.run()
            place_in_dest(r.df_new, destination)

            # Add df to database to store info
            add_df(df)

        else:
            # Invalid entry - show error message to user
            messagebox.showerror('Error', 'Source and/or Destination Entry Invalid, try again')
            run()

    # User exits out of window
    except AttributeError or TypeError:
        i.cancel()
        sys.exit()


run()

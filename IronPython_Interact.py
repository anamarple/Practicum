import clr
import os

# Connect to .net framework
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")

from System.Windows.Forms import *
from System.Drawing import *


# Create the initial wpf form shown to user
class IronPythonForm(Form):
    def __init__(self):
        # Set up labels, text boxes, buttons
        self._labelSource = Label()
        self._labelDest = Label()
        self._txtSource = TextBox()
        self._txtDest = TextBox()
        self._btnOK = Button()
        self._btnSelect1 = Button()
        self._btnSelect2 = Button()

        # Labels
        self._labelSource.Text = 'Source Path'
        self._labelDest.Text = 'Destination Path'
        self._labelSource.Location = Point(0, 25)
        self._labelDest.Location = Point(0, 55)

        # Source Text Box
        self._txtSource.Location = Point(100, 25)
        self._txtSource.Name = 'Source Path'
        self._txtSource.Size = Size(300, 30)
        self._txtSource.TabIndex = 0

        # Destination Text Box
        self._txtDest.Location = Point(100, 50)
        self._txtDest.Name = 'Destination Path'
        self._txtDest.Size = Size(300, 30)
        self._txtDest.TabIndex = 1

        # 'OK' Button
        self._btnOK.Location = Point(200, 200)
        self._btnOK.Text = 'OK'
        self._btnOK.TabIndex = 2
        self._btnOK.Click += self.OK_clicked

        # 'Select' Buttons
        self._btnSelect1.Location = Point(410, 25)
        self._btnSelect1.Text = 'Select'
        self._btnSelect1.Size = Size(50, 22)
        self._btnSelect1.TabIndex = 3
        self._btnSelect1.Click += self.Select1_clicked

        self._btnSelect2.Location = Point(410, 50)
        self._btnSelect2.Text = 'Select'
        self._btnSelect2.Size = Size(50, 22)
        self._btnSelect2.TabIndex = 4
        self._btnSelect2.Click += self.Select2_clicked

        # Form
        self.Controls.Add(self._txtSource)
        self.Controls.Add(self._txtDest)
        self.Controls.Add(self._labelSource)
        self.Controls.Add(self._labelDest)
        self.Controls.Add(self._btnOK)
        self.Controls.Add(self._btnSelect1)
        self.Controls.Add(self._btnSelect2)

        self.Text = 'D&M File Classification Tool'
        self.Height = 300
        self.Width = 500
        self.Icon = Icon(r'T:\DS\DZS\personal\favicon.ico')

    # 'Select' button 1 has been clicked, gives user option to select folder from browser
    def Select1_clicked(self, sender, args):
        dialog = FolderBrowserDialog()
        if dialog.ShowDialog():
            self._txtSource.Text = dialog.SelectedPath

    # 'Select' button 2 has been clicked, gives user option to select folder from browser
    def Select2_clicked(self, sender, args):
        dialog = FolderBrowserDialog()
        if dialog.ShowDialog():
            self._txtDest.Text = dialog.SelectedPath

    # 'OK' button has been clicked
    def OK_clicked(self, sender, args):
        # Making sure text box contents are valid directory paths, then output paths
        if os.path.isdir(self._txtSource.Text) and os.path.isdir(self._txtDest.Text):
            print(self._txtSource.Text)
            print(self._txtDest.Text)
            self.Close()

        # User left input boxes blank and/or gave invalid directory paths
        else:
            MessageBox.Show("Please provide valid Source and/or Destination Path and try again", "Error",
                            MessageBoxButtons.OK, MessageBoxIcon.Error)


form = IronPythonForm()
Application.Run(form)

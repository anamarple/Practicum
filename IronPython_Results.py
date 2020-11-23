import clr
import sys

# Connect to .net framework
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")

from System.Windows.Forms import *
from System.Drawing import *

# Sources:
# https://ericgazoni.wordpress.com/2010/05/03/ironpython-wpf/
# http://www.functionx.com/vcsharp2010/controls/dgv.
# https://mail.python.org/pipermail/ironpython-users/2010-February/012266.html
# https://stackoverflow.com/questions/35940653/create-right-click-menu-on-datagridview-for-entire-row

# Returns data grid to user to view categorizations, allows user to make corrections
class IronPythonResults(Form):

    def __init__(self, _results):
        # Add 'OK' button
        self._btnOK = Button()
        self._btnOK.Location = Point(900, 910)
        self._btnOK.Text = 'OK'
        self._btnOK.Click += self.OK_clicked

        # Add grid
        self.grid = DataGridView()
        self.grid.Columns.Add("Path", "Path")
        self.grid.Columns.Add("File", "File")
        self.grid.Columns.Add("Bucket1", "Bucket1")
        self.grid.Columns.Add("Bucket2", "Bucket2")
        self.grid.Size = Size(1000, 900)

        for path, file, bucket1, bucket2 in _results:
            self.grid.Rows.Add(path, file, bucket1, bucket2)

        self.Controls.Add(self.grid)
        self.Controls.Add(self._btnOK)
        self.grid.SelectionMode = DataGridViewSelectionMode.FullRowSelect
        self.grid.AllowUserToAddRows = False
        self.grid.MouseDown += self.rows_selected

        # TODO: Add 'Export datagrid to csv file' option
        '''
        self.menuStrip = MenuStrip()
        self.menuStrip.Dock = DockStyle.Top
        #self.exportOption = MenuItem("Export to csv", self.export)
        self.menuStrip.Items.Add("Export to csv")
        self.Controls.Add(self.menuStrip)
        '''

        # Form
        self.Text = 'File Classification Results'
        self.Height = 980
        self.Width = 1000
        self.Icon = Icon(r'T:\DS\DZS\personal\favicon.ico')

    # Gives popup menu on right-click for user to change bucket2 of selected cells
    def rows_selected(self, sender, args):
        if args.Button == MouseButtons.Right:

            self.cm = ContextMenu()
            self.changeBucket = MenuItem("Change Bucket2 to...")
            self.cm.MenuItems.Add(self.changeBucket)

            self.prev = MenuItem("Previous", self.change_bucket_prev)
            self.gen = MenuItem("General", self.change_bucket_gen)
            self.vol = MenuItem("Volumetric and Reserves Estimates", self.change_bucket_vol)
            self.prod = MenuItem("Production", self.change_bucket_prod)
            self.dev = MenuItem("Development Plans", self.change_bucket_dev)
            self.econ = MenuItem("Economics", self.change_bucket_econ)
            self.field = MenuItem("Field Reports", self.change_bucket_field)
            self.seis = MenuItem("Seismic Data", self.change_bucket_seis)
            self.geo = MenuItem("Geologic Maps", self.change_bucket_geo)
            self.bubble = MenuItem("Bubble Maps", self.change_bucket_bubble)
            self.pvt = MenuItem("PVT and Test Data", self.change_bucket_pvt)
            self.petro = MenuItem("Petrophysical Summaries", self.change_bucket_petro)
            self.cross = MenuItem("Cross-Sections", self.change_bucket_cross)
            self.logs = MenuItem("Logs", self.change_bucket_logs)
            self.act = MenuItem("Field Activity", self.change_bucket_act)
            self.model = MenuItem("Modeling", self.change_bucket_model)

            commands = [self.prev, self.gen, self.vol, self.prod, self.dev, self.econ,
                        self.field, self.seis, self.geo, self.bubble, self.pvt, self.petro,
                        self.cross, self.logs, self.act, self.model]
            for command in commands:
                self.changeBucket.MenuItems.Add(command)
            self.cm.Show(self.grid, args.Location)

    # 'OK' button has been clicked, return df to driver program
    def OK_clicked(self, sender, args):
        for row in self.grid.Rows:
            for cell in row.Cells:
                print(cell.Value)
        self.Close()

    '''
    # Export datagrid to csv file
    def export(self, sender, args):
        print("export")
    '''

    # Changes data grid cell value to user-defined bucket
    def change_bucket(self, bucket):
        rows = self.grid.SelectedRows
        for row in rows:
            # Update bucket2
            self.grid[3, row.Index].Value = bucket

            # Update bucket2 accordingly
            if bucket == 'Economics':
                self.grid[2, row.Index].Value = "Economics"
            elif bucket == 'Development Plans' or bucket == 'Field Reports' or bucket == 'Field Activity':
                self.grid[2, row.Index].Value = 'Completion'
            elif bucket == 'Production' or bucket == 'PVT and Test Data':
                self.grid[2, row.Index].Value = 'Production'
            elif bucket == 'General' or bucket == 'Presentations' or bucket == 'Previous':
                self.grid[2, row.Index].Value = 'General'
            else:
                self.grid[2, row.Index].Value = 'Geological'

    def change_bucket_prev(self, sender, args):
        self.change_bucket("Previous")

    def change_bucket_gen(self, sender, args):
        self.change_bucket("General")

    def change_bucket_vol(self, sender, args):
        self.change_bucket("Volumetric and Reserves Estimates")

    def change_bucket_prod(self, sender, args):
        self.change_bucket("Production")

    def change_bucket_dev(self, sender, args):
        self.change_bucket("Development Plans")

    def change_bucket_econ(self, sender, args):
        self.change_bucket("Economics")

    def change_bucket_field(self, sender, args):
        self.change_bucket("Field Reports")

    def change_bucket_seis(self, sender, args):
        self.change_bucket("Seismic Data")

    def change_bucket_geo(self, sender, args):
        self.change_bucket("Geologic Maps")

    def change_bucket_bubble(self, sender, args):
        self.change_bucket("Bubble Maps")

    def change_bucket_pvt(self, sender, args):
        self.change_bucket("PVT and Test Data")

    def change_bucket_petro(self, sender, args):
        self.change_bucket("Petrophysical Summaries")

    def change_bucket_cross(self, sender, args):
        self.change_bucket("Cross-Sections")

    def change_bucket_logs(self, sender, args):
        self.change_bucket("Logs")

    def change_bucket_act(self, sender, args):
        self.change_bucket("Field Activity")

    def change_bucket_model(self, sender, args):
        self.change_bucket("Modeling")


# Create list of lists
def readInResults(input):
    elements = input.split(b"*")
    elements = "".join(elements)
    elements = elements.split(b"|")

    df = []
    for element in elements:
        e = element.split(b",")
        e.remove("")
        df.append(e)
    # Remove last item in list
    del df[(len(df)-1)]
    return df


x = readInResults(sys.stdin.read())
Application.EnableVisualStyles()
form = IronPythonResults(x)
Application.Run(form)
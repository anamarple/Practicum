#!c:\users\amarple\pycharmprojects\practicum\venv\scripts\pythonw.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pandastable==0.12.2.post1','gui_scripts','dataexplore'
__requires__ = 'pandastable==0.12.2.post1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pandastable==0.12.2.post1', 'gui_scripts', 'dataexplore')()
    )

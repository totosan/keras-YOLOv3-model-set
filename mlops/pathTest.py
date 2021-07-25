import os
from pathlib import Path

files = Path(".")

[print(f.name) for f in files.iterdir()]

full = os.path.join("./outputs/","outputs/myfile.py")
print(full)
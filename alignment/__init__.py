import os
import sys
from pathlib import PosixPath

DATA_DIR = (
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)
import os
import re
import numpy as np
from glob import glob
from datetime import datetime


def get_files(directory, pattern):
    return np.asarray(sorted(glob(os.path.join(directory, pattern))))


def get_dt():
    return re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')

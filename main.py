import tensorflow as tf
import numpy as np

from utils import data, visual

# r_fpath = "dataset/PET_CT_hdr/4-CT.hdr"
r_fpath = "dataset/output2/3d-pat-Dose.raw"

data.read_raw(r_fpath)

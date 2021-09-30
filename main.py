import tensorflow as tf
import numpy as np

from utils import data, visual

# r_fpath = "dataset/PET_CT_hdr/4-CT.hdr"
r_fpath = "dataset/patient2/dosemap.raw"

dosemap = data.read_raw(r_fpath)
print(dosemap.shape)
print(dosemap.dtype)
print(dosemap.min())
print(dosemap.max())
# visual.imshow3D_opencv(dosemap)



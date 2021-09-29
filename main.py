import tensorflow as tf
import numpy as np

from utils import data, visual

r_fpath = "dataset/PET_CT_hdr/4-CT.hdr"
s_fpath = "dataset/npy/4_CT.npy"
# data.save_ndarray('nii', r_fpath, s_fpath)

# visual.imshow3D_opencv(img, 256, 256, 100)

data.header_info(r_fpath)

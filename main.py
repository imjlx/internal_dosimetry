import tensorflow as tf
import numpy as np

from utils import data, visual

# r_fpath = "dataset/PET_CT_hdr/4-CT.hdr"
'''r_fpath = "dataset/patient4/atlas.hdr"

dosemap = data.read_nib(r_fpath)
print(dosemap.shape)
print(dosemap.dtype)
print(dosemap.min())
print(dosemap.max())
visual.histogram(dosemap, img_type="atlas")'''
# visual.imshow3D_opencv(dosemap, img_type="dosemap")

patient = data.Patient(5)
patient.create_block()

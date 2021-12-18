#create proper file structure
import os
import shutil
from PIL import Image
import numpy as np

# Configure data path for celltracking data set PhC-C2DH-U373 and file structure
path_dataset = "E:/PythonIDEP/CELLSEGMENTATION/PhC-C2DH-U373/"
path_filestructure = "E:/PythonIDEP/CELLSEGMENTATION/DATA"
if not os.path.exists(path_filestructure): os.mkdir(path_filestructure)

# Iterate over both data sets
for ds in ["01", "02"]:
    # Define image directories
    path_ds_img = os.path.join(path_dataset, ds)
    path_ds_seg = os.path.join(path_dataset, ds + "_GT", "SEG")
    # Obtain sample list
    sample_list = os.listdir(path_ds_seg)
    # Remove every file which does not match image typ and preprocess sample names
    for i in reversed(range(0, len(sample_list))):
        if not sample_list[i].endswith(".tif"):
            del sample_list[i]
        else:
            sample_list[i] = sample_list[i][7:]
    # Iterate over each sample and transform the data into desired file structure
    for sample in sample_list:
        index = ds + "_" + sample[:-4]
        # Create sample directory
        path_sampleDir = os.path.join(path_filestructure, index)
        if not os.path.exists(path_sampleDir): os.mkdir(path_sampleDir)
        # Copy image file into filestructure
        path_ds_sample_img = os.path.join(path_ds_img, "t" + sample)
        path_fs_sample_img = os.path.join(path_sampleDir, "imaging.tif")
        shutil.copy(path_ds_sample_img, path_fs_sample_img)
        # Copy segmentation file into filestructure
        seg_file = "man_seg" + sample
        path_ds_sample_seg = os.path.join(path_ds_seg, seg_file)
        path_fs_sample_seg = os.path.join(path_sampleDir, "segmentation.tif")
        # Load segmentation from file
        seg_raw = Image.open(path_ds_sample_seg)
        # Convert segmentation from Pillow image to numpy matrix
        seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        # Keep only intensity and remove maximum intensitiy range
        seg_data = seg[:,:,0]
        # Union all separate cell classes to a single one
        seg_data[seg_data > 0] = 1
        # Transform numpy array back to a Pillow image & save to disk
        seg = Image.fromarray(seg_data)
        seg.save(path_fs_sample_seg, format="TIFF")
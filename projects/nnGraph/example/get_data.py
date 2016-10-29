#!/usr/bin/env python
# Stephen Gould <stephen.gould@anu.edu.au>
#
# Downloads MSRC data ready for nnGraphPipeline.py
#

import glob
import os
import shutil
import urllib.request
import zipfile
from PIL import Image

DARWIN = os.path.join(os.getcwd(), "../../..")
BIN_DIR = os.path.join(DARWIN, "bin")
DATA_DIR = os.path.join(os.getcwd(), "data")

CONFIG = "msrcConfig.xml"
FILE_LIST = "msrcAllList.txt"

# download dataset
FILENAME = "msrc_objcategimagedatabase_v2.zip"
URLBASE = "http://download.microsoft.com/download/3/3/9/339D8A24-47D7-412F-A1E8-1A415BC48A15/"
if not os.path.exists(FILENAME):
    print("Downloading {}...".format(FILENAME))
    urllib.request.urlretrieve(URLBASE + FILENAME, FILENAME)

# extract dataset
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")

if not os.path.exists(DATA_DIR):
    print("Creating data directory...")
    os.makedirs(DATA_DIR)
    os.makedirs(IMG_DIR)
    os.makedirs(LBL_DIR)

    # extract and convert files
    with zipfile.ZipFile(FILENAME) as zf:
        members = zf.namelist()
        for m in members:
            if "Images/" in m and ".bmp" in m:
                imgfilename = os.path.join(IMG_DIR, os.path.basename(m))
                with open(imgfilename, 'wb') as f:
                    f.write(zf.read(m))
                    Image.open(imgfilename).save(imgfilename[:-3] + "jpg")
            if "GroundTruth/" in m and ".bmp" in m:
                with open(os.path.join(LBL_DIR, os.path.basename(m)), 'wb') as f:
                    f.write(zf.read(m))

    # convert labels
    os.system(os.path.join(BIN_DIR, "convertPixelLabels") + " -config " + CONFIG +
              " -i _GT.bmp -o .txt " + FILE_LIST)
    
                    
print("A data directory should now exist with images and labels subdirectories. If")
print("something went wrong try deleting the data directory and run the script again.")

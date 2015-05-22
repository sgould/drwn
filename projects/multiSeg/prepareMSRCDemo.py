#!/usr/bin/python3
# SCRIPT FOR PREPARING MSRC DATASET PIXEL LABELING
# Stephen Gould <stephen.gould@anu.edu.au>
#

import fileinput
import os
import shutil
import sys
import urllib.request
import zipfile

# script configuration -----------------------------------------------

if len(sys.argv) != 2:
    print("USAGE: " + sys.argv[0] + " <experiment directory>")
    exit(0)

EXPR_DIR = sys.argv[1]
DARWIN = os.path.join(os.getcwd(), "../..")
BIN_DIR = os.path.join(DARWIN, "bin")

CONFIG = "msrcConfig.xml"

TRAIN_LIST = "msrcTrainList.txt"
VAL_LIST = "msrcValList.txt"
TEST_LIST = "msrcTestList.txt"

# create experiment directories --------------------------------------

if not os.path.exists(EXPR_DIR):
    os.makedirs(EXPR_DIR)

shutil.copyfile(CONFIG, os.path.join(EXPR_DIR, CONFIG))
shutil.copyfile("pixSegPipeline.py", os.path.join(EXPR_DIR, "pixSegPipeline.py"))
shutil.copyfile(TRAIN_LIST, os.path.join(EXPR_DIR, TRAIN_LIST))
shutil.copyfile(VAL_LIST, os.path.join(EXPR_DIR, VAL_LIST))
shutil.copyfile(TEST_LIST, os.path.join(EXPR_DIR, TEST_LIST))

os.chdir(EXPR_DIR)

# download and preprocess data ---------------------------------------

URL_BASE = "http://research.microsoft.com/en-us/um/people/antcrim/data_objrec/"
URL_FILE = "msrc_objcategimagedatabase_v2.zip"

# fetch data
if (os.path.isfile(URL_FILE) == 0):
    print("WARNING: Downloading images. This make take a while...")
    try:
        urllib.request.urlretrieve(URL_BASE + URL_FILE, URL_FILE)
    except:
        print("ERROR: Could not download data file " + URL_FILE)
        exit()

# extract data
with zipfile.ZipFile(URL_FILE, "r") as z:
    z.extractall()

# convert labels
os.system(os.path.join(BIN_DIR, "convertPixelLabels") + " -config " + CONFIG +
    " -i _GT.bmp -o .txt " + TRAIN_LIST)
os.system(os.path.join(BIN_DIR, "convertPixelLabels") + " -config " + CONFIG +
    " -i _GT.bmp -o .txt " + VAL_LIST)
os.system(os.path.join(BIN_DIR, "convertPixelLabels") + " -config " + CONFIG +
    " -i _GT.bmp -o .txt " + TEST_LIST)

# pixel segmentation model -------------------------------------------

print("--- DATA PREPARATION COMPLETE -----------------------------")
print("Now cd into directory '" + EXPR_DIR + "' and run pixSegPipeline.py")
print("-----------------------------------------------------------")



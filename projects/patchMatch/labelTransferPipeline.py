#!/usr/bin/python3
# Stephen Gould <stephen.gould@anu.edu.au>
#
# This script is setup to run the PatchMatchGraph label transfer
# algorithm on the Stanford Background Dataset. The algorithm is
# described in Gould and Zhang, "PatchMatchGraph: Building a Graph of
# Dense Patch Correspondences for Label Transfer", ECCV 2014
#

import os
import sys
import urllib.request
import tarfile
import random

# script configuration -----------------------------------------------

BIN_DIR = os.path.join(os.getcwd(), "../../bin")

IMAGE_LIST = "stanfordAllList.txt"
CONFIG = "stanfordBGConfig.xml"

DATASET = "stanford"
DATA_DIR = "iccv09Data/images/"

# download data ------------------------------------------------------

if (not os.path.isdir("iccv09Data")):

    URL_BASE = "http://users.cecs.anu.edu.au/~sgould/papers/"
    URL_FILE = "iccv09Data.tar.gz"

    # fetch data
    if (os.path.isfile(URL_FILE) == 0):
        print("Downloading " + URL_FILE)
        try:
            urllib.request.urlretrieve(URL_BASE + URL_FILE, URL_FILE)
        except:
            print("ERROR: could not download data file " + URL_FILE)
            exit()

    # extract data
    tar = tarfile.open(URL_FILE, "r:gz")
    tar.extractall()
    tar.close()

# extract basenames --------------------------------------------------

base_names = [f[:-4] for f in os.listdir("iccv09Data/images/")]
random.shuffle(base_names)

fh = open(IMAGE_LIST, "w")
fh.write("\n".join(base_names) + "\n")
fh.close()

# create output directory --------------------------------------------

if not os.path.exists("output"):
    os.makedirs("output")

# label transfer algorithm -------------------------------------------

# build graph
if True:
    if os.system(os.path.join(BIN_DIR, "buildPatchMatchGraph") + " -config " + CONFIG +
        " -d iccv09Data/images/ -e .jpg -m 50 -o " + DATASET + " " + IMAGE_LIST) != 0:
        exit()

# perform label transfer
if True:
    if os.system(os.path.join(BIN_DIR, "patchMatchLabelTransfer") + " -config " + CONFIG +
        " -outLabels .pm.txt -outImages .pm.png " + DATASET + " " + IMAGE_LIST) != 0:
        exit()

# score results ------------------------------------------------------

os.system(os.path.join(BIN_DIR, "scorePixelLabels") + " -config " + CONFIG +
    " -inLabels .pm.txt " + IMAGE_LIST)

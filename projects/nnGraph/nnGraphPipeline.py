#!/usr/bin/python
# Stephen Gould <stephen.gould@anu.edu.au>
#
# Reference implementation of the semantic segmentation model
# described in Gould et al., "Superpixel Graph Label Transfer with
# Learned Distance Metric", ECCV 2014.
#
# Assumes that directories data/images and data/labels exists with
# JPEG images and text label files, respectively.
#

import glob
import os
import shutil
import sys

# script configuration -----------------------------------------------

DARWIN = os.path.join(os.getcwd(), "../..")
BIN_DIR = os.path.join(DARWIN, "bin")
DATA_DIR = os.path.join(os.getcwd(), "data")

if not os.path.exists(DATA_DIR):
    print("ERROR: data directory does not exist")
    # TODO: prompt user to download and setup data directory
    exit(1)

# dataset configuration ----------------------------------------------

NN_CONFIG = "nnGraphConfig.xml"

DS_NAME = "nnMSRC"
DS_CONFIG = "msrcConfig.xml"
DS_TRAIN_LIST = "msrcTrainValList.txt"
DS_TEST_LIST = "msrcTestList.txt"

LOG_FILE = DS_NAME + ".log"

# command line options for darwin applications -----------------------

STD_OPTIONS = " -verbose -profile -log " + DS_NAME + ".log -threads 8"

# function to construct command line
def cmdline(exe, args):
    return os.path.join(BIN_DIR, exe) + " " + str(" ").join(args)

# create output directory --------------------------------------------

if not os.path.exists("output"):
    os.makedirs("output")

# create superpixels -------------------------------------------------

if True:
    SEG_DIR = os.path.join(DATA_DIR, "regions")
    if not os.path.exists(SEG_DIR):
        os.makedirs(SEG_DIR)

    for img_file in glob.glob(os.path.join(DATA_DIR, "images/*.jpg")):
        basename = os.path.basename(img_file)[:-4]
        seg_file = os.path.join(SEG_DIR, basename + ".bin")
        if not os.path.exists(seg_file):
            args = ["-verbose -log", LOG_FILE, "-m SUPERPIXEL",
                 "-g 24 -g 17 -g 12 -g 8 -g 6 -g 4", "-o", seg_file, img_file]
            os.system(cmdline("generateSuperpixels", args))

# initialize graph with training images ------------------------------

if not os.path.exists(DS_NAME + ".init_train.index"):
    args = [STD_OPTIONS, "-config", NN_CONFIG,
            " -o ", DS_NAME + ".init_train", DS_TRAIN_LIST]
    os.system(cmdline("nnGraphInitialize", args))

if not os.path.exists(DS_NAME + ".init_all.index"):
    args = [STD_OPTIONS, "-config", NN_CONFIG, "-i", DS_NAME + ".init_train", 
            "-o", DS_NAME + ".init_all", DS_TEST_LIST]
    os.system(cmdline("nnGraphInitialize", args))

# experiments subroutine ---------------------------------------------

def run_experiment(k, tag, xform = None):

    EXPRTAG = tag + "_" + str(k)
    GRAPH = DS_NAME + "_" + EXPRTAG
    OPTIONS = STD_OPTIONS + " -config " + NN_CONFIG + " -set drwnNNGraph K " + str(k)

    # learn and apply transform
    if (not xform is None):
        args = [xform, "-o", GRAPH + ".xform", DS_NAME + ".init_train"]
        os.system(cmdline("nnGraphLearnTransform", args))
        args = [OPTIONS, "-o", GRAPH, GRAPH + ".xform", DS_NAME + ".init_all"]
        os.system(cmdline("nnGraphApplyTransform", args))
    else:
        shutil.copyfile(DS_NAME + ".init_all.data", GRAPH + ".data")
        shutil.copyfile(DS_NAME + ".init_all.index", GRAPH + ".index")

    # build graph
    args = [OPTIONS, "-m 50", "-i", GRAPH,  "-o", GRAPH, "-not", DS_TEST_LIST, DS_TRAIN_LIST]
    os.system(cmdline("nnGraphOptimize", args))
    args = [OPTIONS, "-m 50", "-i", GRAPH,  "-o", GRAPH, "-eqv", DS_TEST_LIST, DS_TEST_LIST]
    os.system(cmdline("nnGraphOptimize", args))

    # transfer labels and score
    args = ["-config", DS_CONFIG, OPTIONS, "-outImages",  "'" + EXPRTAG + ".png'",
            "-outLabels", "'" + EXPRTAG + ".txt'", GRAPH, DS_TEST_LIST]
    os.system(cmdline("nnGraphLabelTransfer", args))
    args = ["-config", DS_CONFIG, STD_OPTIONS, "-inLabels", "'" + EXPRTAG + ".txt'", DS_TEST_LIST]
    os.system(cmdline("scorePixelLabels", args))

# experiments --------------------------------------------------------

# euclidean distance
for k in [1, 2, 5, 10, 15, 20]:
    run_experiment(k, "euclid")

# diagonal mahalanobis distance
for k in [1, 2, 5, 10, 15, 20]:
    run_experiment(k, "white", "-t Whitener")

# learned large margin nearest neighbour distance
for k in [1, 2, 5, 10, 15, 20]:
    run_experiment(k, "white", "-t LMNN -m 5")

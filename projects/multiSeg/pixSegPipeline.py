#!/usr/bin/python
# Stephen Gould <stephen.gould@anu.edu.au>
#
# This script is setup to run on the 21-class MSRC dataset. It will
# require about 4GB of memory and about 5GB of diskspace if caching
# is enabled. Accuracy should be around 79.0% for the unary model
# and 84.5% for the CRF model on the 256 images in the test set. The
# script assumes that the data has been appropriately pre-processed
# (see prepareMSRCDemo.py).
#

import os
import sys

# script configuration -----------------------------------------------

BIN_DIR = os.path.join(os.getcwd(), "../../../bin")

TRAIN_LIST = "msrcTrainList.txt"
VAL_LIST = "msrcValList.txt"
TEST_LIST = "msrcTestList.txt"

CONFIG = "msrcConfig.xml"

# change to experiments directory ------------------------------------

# create directories
if not os.path.exists("cached"):
    os.makedirs("cached")

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("output"):
    os.makedirs("output")

# pixel segmentation model -------------------------------------------

# train boosted classifiers
if True:
    map(os.unlink, [os.path.join("cached", f) for f in os.listdir("cached")])

    if os.system(os.path.join(BIN_DIR, "learnPixelSegModel") + " -config " + CONFIG + " -component BOOSTED" +
        " -set drwnDecisionTree split MISCLASS" +
        " -set drwnBoostedClassifier numRounds 200" +
        " -subSample 250 " + TRAIN_LIST) != 0:
        exit()

# train unary potentials
if True:
    if os.system(os.path.join(BIN_DIR, "learnPixelSegModel") + " -config " + CONFIG + " -component UNARY" +
        " -subSample 25 " + TRAIN_LIST) != 0:
        exit()

# evaluate test set on unary potentials
if True:
    if os.system(os.path.join(BIN_DIR, "inferPixelLabels") + " -config " + CONFIG + " -pairwise 0.0 -longrange 0.0" +
        " -outLabels .unary.txt -outImages .unary.png " + TEST_LIST) != 0:
        exit()

# cross-validate pairwise cost
if True:
    if os.system(os.path.join(BIN_DIR, "learnPixelSegModel") + " -config " + CONFIG + " -component CONTRAST " + 
        VAL_LIST) != 0:
        exit()

# evaluate test set on contrast pairwise model
if True:
    if os.system(os.path.join(BIN_DIR, "inferPixelLabels") + " -config " + CONFIG + " -longrange 0.0" +
        " -outLabels .pairwise.txt -outImages .pairwise.png " + TEST_LIST) != 0:
        exit()

# cross-validate contrast and long-range pairwise cost
if True:
    if os.system(os.path.join(BIN_DIR, "learnPixelSegModel") + " -config " + CONFIG + " -component CONTRASTANDLONGRANGE " + 
        VAL_LIST) != 0:
        exit()

# evaluate test set on contrast and long-range pairwise model
if True:
    if os.system(os.path.join(BIN_DIR, "inferPixelLabels") + " -config " + CONFIG +
        " -outLabels .longrange.txt -outImages .longrange.png " + TEST_LIST) != 0:
        exit()

# score results ------------------------------------------------------

os.system(os.path.join(BIN_DIR, "scorePixelLabels") + " -config " + CONFIG +
    " -inLabels .unary.txt " + TEST_LIST)

os.system(os.path.join(BIN_DIR, "scorePixelLabels") + " -config " + CONFIG +
    " -inLabels .pairwise.txt " + TEST_LIST)

os.system(os.path.join(BIN_DIR, "scorePixelLabels") + " -config " + CONFIG +
    " -inLabels .longrange.txt " + TEST_LIST)

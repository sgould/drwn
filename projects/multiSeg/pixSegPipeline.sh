#!/bin/csh
# Stephen Gould <stephen.gould@anu.edu.au>
#
# This script is setup to run on the 21-class MSRC dataset. It will
# require about 4GB of memory and about 5GB of diskspace if caching
# is enabled. Accuracy should be around 79.0% for the unary model
# and 84.5% for the CRF model on the 256 images in the test set. The
# script assumes that the data has been appropriately pre-processed
# (see prepareMSRCDemo.sh).
#

# script configuration -----------------------------------------------

set BIN_DIR = "${DARWIN}/bin/"

set TRAIN_LIST = "msrcTrainList.txt"
set VAL_LIST = "msrcValList.txt"
set TEST_LIST = "msrcTestList.txt"

set CONFIG = "msrcConfig.xml"

# change to experiments directory ------------------------------------

# create directories
mkdir -p cached
mkdir -p models
mkdir -p output

# pixel segmentation model -------------------------------------------

# train boosted classifiers
if (1) then
    rm -f cached/*

    ${BIN_DIR}/learnPixelSegModel -config $CONFIG -component BOOSTED \
        -set drwnDecisionTree split MISCLASS \
        -set drwnBoostedClassifier numRounds 200 \
        -subSample 250 $TRAIN_LIST || exit 1
endif

# train unary potentials
if (1) then
    ${BIN_DIR}/learnPixelSegModel -config $CONFIG -component UNARY \
        -subSample 25 $TRAIN_LIST || exit 1
endif

# evaluate test set on unary potentials
if (1) then
    ${BIN_DIR}/inferPixelLabels -config $CONFIG -pairwise 0.0 \
        -outLabels .unary.txt -outImages .unary.png \
        $TEST_LIST
endif

# cross-validate pairwise cost
if (1) then
    ${BIN_DIR}/learnPixelSegModel -config $CONFIG -component CONTRAST \
        $VAL_LIST || exit 1
endif

# evaluate test set on pairwise model
if (1) then
    ${BIN_DIR}/inferPixelLabels -config $CONFIG \
        -outLabels .pairwise.txt -outImages .pairwise.png \
        $TEST_LIST
endif

# score results ------------------------------------------------------

${BIN_DIR}/scorePixelLabels -config $CONFIG \
    -inLabels .unary.txt $TEST_LIST

${BIN_DIR}/scorePixelLabels -config $CONFIG -confusion \
    -inLabels .pairwise.txt $TEST_LIST

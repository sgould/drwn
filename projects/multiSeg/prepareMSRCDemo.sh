#!/bin/csh
# SCRIPT FOR PREPARING MSRC DATASET PIXEL LABELING
# Stephen Gould <stephen.gould@anu.edu.au>
#

# script configuration -----------------------------------------------

if ($#argv != 1) then
    echo "USAGE: $0 <expriment directory>"
    exit 0
endif

set EXPR_DIR = $1
set DARWIN = "`pwd`/../../"
set BIN_DIR = "${DARWIN}/bin/"

set CONFIG = "`pwd`/msrcConfig.xml"

set TRAIN_LIST = "msrcTrainList.txt"
set VAL_LIST = "msrcValList.txt"
set TEST_LIST = "msrcTestList.txt"

# create experiment directories --------------------------------------

mkdir -p $EXPR_DIR

mkdir -p ${EXPR_DIR}/data
mkdir -p ${EXPR_DIR}/data/images
mkdir -p ${EXPR_DIR}/data/labels

cp msrcConfig.xml ${EXPR_DIR}/msrcConfig.xml
cp pixSegPipeline.sh ${EXPR_DIR}/pixSegPipeline.sh
cp ${TRAIN_LIST} ${EXPR_DIR}/${TRAIN_LIST}
cp ${VAL_LIST} ${EXPR_DIR}/${VAL_LIST}
cp ${TEST_LIST} ${EXPR_DIR}/${TEST_LIST}

cd ${EXPR_DIR}

# download and preprocess data ---------------------------------------

set URL_BASE = "http://research.microsoft.com/en-us/um/people/antcrim/data_objrec/"
set URL_FILE = "msrc_objcategimagedatabase_v2.zip"

# fetch data
if (! -e ${URL_FILE}) then
    wget ${URL_BASE}${URL_FILE}
endif

# extract data and basenames
unzip ${URL_FILE}
ls MSRC_ObjCategImageDatabase_v2/Images/*.bmp | sed 's/.*\///' | sed 's/.bmp//' > $ALL_LIST

# convert images
foreach i (`cat $TRAIN_LIST $VAL_LIST $TEST_LIST`)
    convert MSRC_ObjCategImageDatabase_v2/Images/${i}.bmp data/images/${i}.jpg
    mv MSRC_ObjCategImageDatabase_v2/GroundTruth/${i}_GT.bmp data/labels/${i}.bmp
end

# convert labels
${BIN_DIR}/convertPixelLabels -config $CONFIG -i .bmp -o .txt $TRAIN_LIST
${BIN_DIR}/convertPixelLabels -config $CONFIG -i .bmp -o .txt $VAL_LIST
${BIN_DIR}/convertPixelLabels -config $CONFIG -i .bmp -o .txt $TEST_LIST

# pixel segmentation model -------------------------------------------

echo "--- DATA PREPARATION COMPLETE -----------------------------"
echo "Now cd into ${EXPR_DIR} and run ./pixSegPipeline.sh"
echo "-----------------------------------------------------------"

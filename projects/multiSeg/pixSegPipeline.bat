@echo off
REM PIXSEGPIPELINE
REM Stephen Gould <stephen.gould@anu.edu.au>
REM
REM This script is setup to run on the 21-class MSRC dataset. It will
REM require about 4GB of memory and about 5GB of diskspace if caching
REM is enabled. Accuracy should be around 79.0% for the unary model
REM and 84.5% for the CRF model on the 256 images in the test set. The
REM script assumes that the data has been appropriately pre-processed
REM (see online documentation at http://drwn.anu.edu.au).

REM script configuration ---------------------------------------------------

set BIN_DIR=..\..\bin\

set TRAIN_LIST=msrcTrainList.txt
set VAL_LIST=msrcValList.txt
set TEST_LIST=msrcTestList.txt

set CONFIG=msrcConfigWin32.xml

REM experiment setup -------------------------------------------------------

REM create directories
mkdir cached
mkdir models
mkdir output

REM pixel segmentation model -----------------------------------------------

REM train boosted classifiers
if 1 == 1 (
    del cached\*
    %BIN_DIR%learnPixelSegModel -config %CONFIG% -component BOOSTED ^
        -set drwnDecisionTree split MISCLASS ^
        -set drwnBoostedClassifier numRounds 200 ^
        -subSample 250 %TRAIN_LIST% || exit /b
)

REM train unary potentials
if 1 == 1 (
    %BIN_DIR%learnPixelSegModel -config %CONFIG% -component UNARY ^
        -subSample 25 %TRAIN_LIST% || exit /b
)

REM evaluate test set on unary potentials
if 1 == 1 (
    %BIN_DIR%inferPixelLabels -config %CONFIG% -pairwise 0.0 ^
        -outLabels .unary.txt -outImages .unary.png ^
        %TEST_LIST% || exit /b
)

REM cross-validate pairwise cost
if 1 == 1 (
    %BIN_DIR%learnPixelSegModel -config %CONFIG% -component CONTRAST ^
        %VAL_LIST% || exit /b
)

REM evaluate test set on pairwise model
if 1 == 1 (
    %BIN_DIR%inferPixelLabels -config %CONFIG% ^
        -outLabels .pairwise.txt -outImages .pairwise.png ^
        %TEST_LIST%
)

REM score results ------------------------------------------------------

%BIN_DIR%scorePixelLabels -config %CONFIG% ^
    -inLabels .unary.txt %TEST_LIST%

%BIN_DIR%scorePixelLabels -config %CONFIG% -confusion ^
    -inLabels .pairwise.txt %TEST_LIST%



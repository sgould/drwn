# DARWIN LIBRARY MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..

-include $(DRWN_PATH)/make.mk

# library source files ---------------------------------------------------

BASE_LIB_SRC = drwnBitArray.cpp drwnCodeProfiler.cpp drwnConfigManager.cpp \
	drwnFileUtils.cpp drwnInterfaces.cpp drwnLogger.cpp \
	drwnProperties.cpp drwnStatsUtils.cpp drwnStrUtils.cpp drwnThreadPool.cpp \
	drwnXMLParser.cpp drwnXMLUtils.cpp

IO_LIB_SRC = drwnCompressionBuffer.cpp drwnIOUtils.cpp drwnPersistentStorage.cpp

ML_LIB_SRC = drwnBoostedClassifier.cpp drwnClassifier.cpp \
	drwnClassificationResults.cpp drwnCompositeClassifier.cpp \
	drwnConfusionMatrix.cpp drwnCrossValidator.cpp drwnDataset.cpp \
	drwnDecisionTree.cpp drwnDisjointSets.cpp drwnFeatureTransform.cpp \
	drwnFeatureWhitener.cpp drwnFisherLDA.cpp drwnGaussian.cpp drwnGaussianMixture.cpp \
	drwnKMeans.cpp drwnLinearRegressor.cpp drwnLinearTransform.cpp drwnLPSolver.cpp \
	drwnMultiClassLogistic.cpp drwnOptimizer.cpp drwnPCA.cpp drwnQPSolver.cpp \
	drwnRandomForest.cpp drwnRegression.cpp drwnSuffStats.cpp

PGM_LIB_SRC = drwnADLPInference.cpp drwnFactorGraph.cpp drwnGraphCutInference.cpp \
	drwnGraphUtils.cpp drwnInference.cpp drwnMapInference.cpp drwnMapInferenceFactory.cpp \
	drwnMaxFlow.cpp drwnTableFactor.cpp drwnTableFactorOps.cpp \
	drwnTRWSInference.cpp drwnVarAssignment.cpp drwnVarUniverse.cpp

VISION_LIB_SRC = drwnColourHistogram.cpp drwnFilterBankResponse.cpp drwnGrabCutInstance.cpp \
	drwnHOGFeatures.cpp drwnImageCache.cpp drwnImagePyramidCache.cpp \
	drwnInPaint.cpp drwnLBPFilterBank.cpp drwnMultiSegConfig.cpp drwnMultiSegVis.cpp \
	drwnObject.cpp drwnOpenCVUtils.cpp drwnPatchMatch.cpp drwnPatchMatchUtils.cpp \
	drwnPartsModel.cpp drwnPixelNeighbourContrasts.cpp drwnPixelSegCRFInference.cpp \
	drwnPixelSegModel.cpp drwnSegImageInstance.cpp drwnSegImagePixelFeatures.cpp \
	drwnSegImageRegionFeatures.cpp drwnSuperpixelContainer.cpp \
	drwnTemplateMatcher.cpp drwnTextonFilterBank.cpp drwnVisionUtils.cpp

# directory for .o and .d files
TEMP_DIR = _intermediates

# library subdirectories
LIB_SRC_DIRS = base io ml pgm vision

# ------------------------------------------------------------------------

BASE_LIB_OBJ = ${addprefix ${TEMP_DIR}/base/,$(BASE_LIB_SRC:.cpp=.o)}
IO_LIB_OBJ  = ${addprefix ${TEMP_DIR}/io/,$(IO_LIB_SRC:.cpp=.o)}
ML_LIB_OBJ  = ${addprefix ${TEMP_DIR}/ml/,$(ML_LIB_SRC:.cpp=.o)}
PGM_LIB_OBJ  = ${addprefix ${TEMP_DIR}/pgm/,$(PGM_LIB_SRC:.cpp=.o)}
VISION_LIB_OBJ  = ${addprefix ${TEMP_DIR}/vision/,$(VISION_LIB_SRC:.cpp=.o)}

LIB_OBJ = ${BASE_LIB_OBJ} ${IO_LIB_OBJ} ${ML_LIB_OBJ} ${PGM_LIB_OBJ}

CFLAGS += -I$(SRC_PATH)/lib/base -I$(SRC_PATH)/lib/io -I$(SRC_PATH)/lib/ml -I$(SRC_PATH)/lib/pgm

EXTRA_IO_LIBS = 
ifeq ($(DRWN_BUILD_WITH_ZLIB), 1)  
	EXTRA_IO_LIBS += $(EXT_PATH)/zlib/libz.a
endif

ifeq ($(DRWN_BUILD_VISION_LIB), 1)
  CFLAGS += -I$(SRC_PATH)/lib/vision
  LIB_OBJ += ${VISION_LIB_OBJ}
endif

DEPS = ${LIB_OBJ:.o=.d}

.PHONY: clean
.PRECIOUS: $(LIB_OBJ)

all: temp_dirs $(STATICDRWN) $(SHAREDDRWN)

temp_dirs:
	for subdir in ${LIB_SRC_DIRS}; do mkdir -p ${TEMP_DIR}/$$subdir; done

# static libraries
$(BIN_PATH)/libdrwnBase.a: $(BASE_LIB_OBJ)
	ar rcs $@ $(BASE_LIB_OBJ)

$(BIN_PATH)/libdrwnIO.a: $(IO_LIB_OBJ)
	ar rcs $@ $(IO_LIB_OBJ)

$(BIN_PATH)/libdrwnML.a: $(ML_LIB_OBJ)
	ar rcs $@ $(ML_LIB_OBJ)

$(BIN_PATH)/libdrwnPGM.a: $(PGM_LIB_OBJ)
	ar rcs $@ $(PGM_LIB_OBJ)

$(BIN_PATH)/libdrwnVision.a: $(VISION_LIB_OBJ)
	ar rcs $@ $(VISION_LIB_OBJ)

# shared libraries (Linux)
$(BIN_PATH)/libdrwnBase.so.${DRWNLIBVER}: $(BASE_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnBase.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnBase.so.${DRWNLIBVER} $(BASE_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

$(BIN_PATH)/libdrwnIO.so.${DRWNLIBVER}: $(IO_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnIO.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnIO.so.${DRWNLIBVER} $(IO_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

$(BIN_PATH)/libdrwnML.so.${DRWNLIBVER}: $(ML_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnML.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnML.so.${DRWNLIBVER} $(ML_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

$(BIN_PATH)/libdrwnPGM.so.${DRWNLIBVER}: $(PGM_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnPGM.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnPGM.so.${DRWNLIBVER} $(PGM_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

$(BIN_PATH)/libdrwnVision.so.${DRWNLIBVER}: $(VISION_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnVision.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnVision.so.${DRWNLIBVER} $(VISION_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

# shared libraries (Mac OS X)
$(BIN_PATH)/libdrwnBase.${DRWNLIBVER}.dylib: $(BASE_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lstdc++ - $(BASE_LIB_OBJ)

$(BIN_PATH)/libdrwnIO.${DRWNLIBVER}.dylib: $(IO_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lstdc++ $(EXTRA_IO_LIBS) $(BIN_PATH)/libdrwnBase.${DRWNLIBVER}.dylib - $(IO_LIB_OBJ)

$(BIN_PATH)/libdrwnML.${DRWNLIBVER}.dylib: $(ML_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lstdc++ $(BIN_PATH)/libdrwnBase.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnIO.${DRWNLIBVER}.dylib - $(ML_LIB_OBJ)

$(BIN_PATH)/libdrwnPGM.${DRWNLIBVER}.dylib: $(PGM_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lstdc++ $(BIN_PATH)/libdrwnBase.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnIO.${DRWNLIBVER}.dylib - $(PGM_LIB_OBJ)

$(BIN_PATH)/libdrwnVision.${DRWNLIBVER}.dylib: $(VISION_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lstdc++ $(BIN_PATH)/libdrwnBase.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnIO.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnML.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnPGM.${DRWNLIBVER}.dylib `pkg-config --libs ${EXT_PATH}/opencv/lib/pkgconfig/opencv.pc` - $(VISION_LIB_OBJ)

# dependencies and object files
${TEMP_DIR}/%.o : %.cpp
	$(CCC) -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
	${CCC} ${CFLAGS} -c $< -o $@

# clean
clean:
	-rm $(LIB_OBJ)
	-rm $(STATICDRWN)
	-rm $(SHAREDDRWN)
	-rm $(DEPS)

-include $(DEPS)

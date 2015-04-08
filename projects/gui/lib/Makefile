# DARWIN LIBRARY MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../../..
PROJ_PATH := $(shell pwd)/..

-include $(DRWN_PATH)/make.mk

# library source files ---------------------------------------------------

ENGINE_LIB_SRC = drwnDatabase.cpp drwnDataRecord.cpp drwnExporter.cpp \
	drwnGraph.cpp drwnGUIWindow.cpp drwnNode.cpp drwnPort.cpp \
	drwnNodeFactory.cpp drwnDebuggingNodes.cpp

NODES_LIB_SRC = drwnBinaryOpNode.cpp drwnBoostingNode.cpp drwnCodebookNodes.cpp \
	drwnConcatenationNode.cpp drwnConfusionMatrixNode.cpp \
	drwnDataExplorerNode.cpp drwnDecisionTreeNode.cpp drwnDitherPlotNode.cpp \
	drwnImportExportFilesNode.cpp drwnLinearRegressionNode.cpp \
	drwnLinearTransformNodes.cpp drwnMultiClassLogisticNode.cpp \
	drwnPRCurveNode.cpp drwnRollupNode.cpp drwnScatterPlotNode.cpp \
	drwnTextFileNodes.cpp drwnUnaryOpNode.cpp

# directory for .o and .d files
TEMP_DIR = _intermediates

# library subdirectories
LIB_SRC_DIRS = engine nodes

# ------------------------------------------------------------------------

ENGINE_LIB_OBJ = ${addprefix ${TEMP_DIR}/engine/,$(ENGINE_LIB_SRC:.cpp=.o)}
NODES_LIB_OBJ = ${addprefix ${TEMP_DIR}/nodes/,$(NODES_LIB_SRC:.cpp=.o)}

LIB_OBJ = ${ENGINE_LIB_OBJ} ${NODES_LIB_OBJ}
ifeq ($(DRWN_SHARED_LIBS), 1)
  ifeq ($(OS), Darwin)
    LIBS = $(BIN_PATH)/libdrwnEngine.${DRWNLIBVER}.dylib $(BIN_PATH)/libdrwnNodes.${DRWNLIBVER}.dylib
  else
    LIBS = $(BIN_PATH)/libdrwnEngine.so.${DRWNLIBVER} $(BIN_PATH)/libdrwnNodes.so.${DRWNLIBVER}
  endif
else
    LIBS = $(BIN_PATH)/libdrwnEngine.a $(BIN_PATH)/libdrwnNodes.a
endif
CFLAGS += -I$(PROJ_PATH)/include -I$(PROJ_PATH)/lib/engine -I$(PROJ_PATH)/lib/nodes

DEPS = ${LIB_OBJ:.o=.d}

.PHONY: clean
.PRECIOUS: $(LIB_OBJ)

all: temp_dirs $(LIBS)

temp_dirs:
	for subdir in ${LIB_SRC_DIRS}; do mkdir -p ${TEMP_DIR}/$$subdir; done

# static libraries
$(BIN_PATH)/libdrwnEngine.a: $(ENGINE_LIB_OBJ)
	ar rcs $@ $(ENGINE_LIB_OBJ)

$(BIN_PATH)/libdrwnNodes.a: $(NODES_LIB_OBJ)
	ar rcs $@ $(NODES_LIB_OBJ)

# shared libraries (Linux)
$(BIN_PATH)/libdrwnEngine.so.${DRWNLIBVER}: $(ENGINE_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnEngine.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnEngine.so.${DRWNLIBVER} $(ENGINE_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

$(BIN_PATH)/libdrwnNodes.so.${DRWNLIBVER}: $(NODES_LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,libdrwnNodes.so.${DRWNLIBMAJORVER} -o $(BIN_PATH)/libdrwnNodes.so.${DRWNLIBVER} $(NODES_LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

# shared libraries (Mac OS X)
$(BIN_PATH)/libdrwnEngine.${DRWNLIBVER}.dylib: $(ENGINE_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lc++ ${SHAREDDRWN} $(WX_LFLAGS) - $(ENGINE_LIB_OBJ)

$(BIN_PATH)/libdrwnNodes.${DRWNLIBVER}.dylib: $(NODES_LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${DRWNLIBVER} -L/usr/lib -lm -lpthread -lc++ ${SHAREDDRWN} $(WX_LFLAGS) $(BIN_PATH)/libdrwnEngine.${DRWNLIBVER}.dylib - $(NODES_LIB_OBJ)

# dependencies and object files
${TEMP_DIR}/%.o : %.cpp
	${CCC} -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
	${CCC} ${CFLAGS} -c $< -o $@

# clean
clean:
	-rm $(LIB_OBJ)
	-rm $(BIN_PATH)/libdrwnNodes.a $(BIN_PATH)/libdrwnEngine.a
	-rm $(BIN_PATH)/libdrwnNodes.so.* $(BIN_PATH)/libdrwnEngine.so.*
	-rm $(BIN_PATH)/libdrwnNodes.*.dylib $(BIN_PATH)/libdrwnEngine.*.dylib
	-rm $(DEPS)

-include $(DEPS)

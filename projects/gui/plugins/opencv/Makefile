# DARWIN LIBRARY PLUGIN MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../../../..
PROJ_PATH := $(shell pwd)/../..
DRWN_PLUGIN = 1

-include $(DRWN_PATH)/make.mk

# plugin source files ----------------------------------------------------

PLUGIN_NAME = OpenCV
PLUGIN_MAJORVER = 1
PLUGIN_MINORVER = 0.0

PLUGIN_SRC = drwnOpenCVNodes.cpp drwnOpenCVUtils.cpp

# ------------------------------------------------------------------------

PLUGIN_LIB = libdrwn${PLUGIN_NAME}Plugin.so
PLUGIN_VER = ${PLUGIN_MAJORVER}.${PLUGIN_MINORVER}

PLUGIN_OBJ = $(PLUGIN_SRC:.cpp=.o)
PLUGIN_DEP = ${PLUGIN_OBJ:.o=.d}

CFLAGS += -I$(PROJ_PATH)/include -I$(EXT_PATH)/opencv/include/opencv
LFLAGS += -L$(EXT_PATH)/opencv/lib -lhighgui -lcv -lcxcore -lml

.PHONY: clean
.PRECIOUS: $(PLUGIN_OBJ)

all: $(BIN_PATH)/${PLUGIN_LIB}.${PLUGIN_VER}

# shared libraries
$(BIN_PATH)/${PLUGIN_LIB}.${PLUGIN_VER}: $(PLUGIN_OBJ)
	${CCC} ${CFLAGS} ${LFLAGS} -shared -Wl,-soname,${PLUGIN_LIB}.${PLUGIN_MAJORVER} -o $(BIN_PATH)/${PLUGIN_LIB}.${PLUGIN_VER} $(PLUGIN_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

# dependencies and object files
%.o : %.cpp
	${CCC} -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
	${CCC} ${CFLAGS} -c $< -o $@

# clean
clean:
	-rm $(PLUGIN_OBJ)
	-rm $(BIN_PATH)/${PLUGIN_LIB}.*
	-rm $(PLUGIN_DEP)

-include $(PLUGIN_DEP)

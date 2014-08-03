# DARWIN PROJECT MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..

#######################################################################
# define project parameters here
#######################################################################

PROJ_NAME = nnGraph
PROJ_MAJORVER = 1
PROJ_MINORVER = 0.0

-include $(DRWN_PATH)/make.mk

#######################################################################
# add project source files here
#######################################################################

APP_SRC = nnGraphInitialize.cpp nnGraphOptimize.cpp nnGraphLabelTransfer.cpp \
	nnGraphLearnTransform.cpp nnGraphApplyTransform.cpp nnGraphAppendFeatures.cpp \
	nnGraphRelabelNodes.cpp nnGraphOnlineLabelTransfer.cpp nnGraphFLANN.cpp

LIB_SRC = drwnNNGraph.cpp drwnNNGraphVis.cpp drwnNNGraphLearn.cpp

#######################################################################

APP_PROG_NAMES = $(APP_SRC:.cpp=)
MEX_PROG_NAMES = $(MEX_SRC:.cpp=.${MEXEXT})
APP_OBJ = $(APP_SRC:.cpp=.o)
LIB_OBJ = $(LIB_SRC:.cpp=.o)
PROJ_DEP = ${APP_OBJ:.o=.d} ${LIB_OBJ:.o=.d}

LIB_VER = ${PROJ_MAJORVER}.${PROJ_MINORVER}

# TODO: fix up for empty library
ifeq (,${LIB_SRC})
  LIB_NAMES =
else
  LIB_NAMES = lib${PROJ_NAME}.a

  ifeq ($(OS), Darwin)
    LIB_NAMES += lib${PROJ_NAME}.${LIB_VER}.dylib
  else
    LIB_NAMES += lib${PROJ_NAME}.so.${LIB_VER}
  endif

  ifeq ($(DRWN_SHARED_LIBS), 0)
    PROJLIB = $(BIN_PATH)/lib${PROJ_NAME}.a
  else
    ifeq ($(OS), Darwin)
      PROJLIB = $(BIN_PATH)/lib${PROJ_NAME}.${LIB_VER}.dylib
    else
      PROJLIB = $(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER}
    endif
    LFLAGS += -l${PROJ_NAME}
  endif
endif

.PHONY: clean lib mex
.PRECIOUS: $(APP_OBJ) $(LIB_OBJ)

all: lib ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
lib: ${addprefix ${BIN_PATH}/,$(LIB_NAMES)}
mex: ${addprefix ${BIN_PATH}/,$(MEX_PROG_NAMES)}

# applications
$(BIN_PATH)/%: %.o $(PROJLIB) $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(PROJLIB) $(LFLAGS)

# mex applications
$(BIN_PATH)/%.${MEXEXT}: %.cpp $(PROJLIB) $(LIBDRWN)
	mex -v -g $*.cpp -o $(@:.cpp=) $(MEXFLAGS) $(PROJLIB) $(LIBDRWN)

# static libraries
$(BIN_PATH)/lib${PROJ_NAME}.a: $(LIB_OBJ)
	ar rcs $@ $(LIB_OBJ)

# shared libraries (linux)
$(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER}: $(LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,lib${PROJ_NAME}.so.${PROJ_MAJORVER} -o $(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER} $(LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

# shared libraries (mac os x)
$(BIN_PATH)/lib${PROJ_NAME}.${LIB_VER}.dylib: $(LIB_OBJ)
	libtool -dynamic -o $@ -current_version ${LIB_VER} -L/usr/lib -lm -lpthread -lstdc++ \
	`pkg-config --libs ${EXT_PATH}/opencv/lib/pkgconfig/opencv.pc` ${SHAREDDRWN} - $(LIB_OBJ)

# darwin libraries
$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

# dependencies and object files
%.o : %.cpp
	${CCC} -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
	${CCC} ${CFLAGS} -c $< -o $@

# clear
clean:
	-rm $(APP_OBJ)
	-rm $(LIB_OBJ)
	-rm ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
	-rm ${addprefix ${BIN_PATH}/,$(LIB_NAMES)}
	-rm ${PROJ_DEP}
	-rm *~

-include ${PROJ_DEP}

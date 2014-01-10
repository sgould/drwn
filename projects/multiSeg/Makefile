# DARWIN PROJECT MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..

#######################################################################
# define project parameters here
#######################################################################

PROJ_NAME = multiSeg
PROJ_MAJORVER = 1
PROJ_MINORVER = 0.0

-include $(DRWN_PATH)/make.mk

#######################################################################
# add project source files here
#######################################################################

APP_SRC = convertPixelLabels.cpp convertFromLabelMe.cpp \
	learnPixelSegModel.cpp inferPixelLabels.cpp \
	scorePixelLabels.cpp viewPixelLabels.cpp viewPixelFeatures.cpp \
	viewColorLegend.cpp

LIB_SRC =

#######################################################################

APP_PROG_NAMES = $(APP_SRC:.cpp=)
APP_OBJ = $(APP_SRC:.cpp=.o)
LIB_OBJ = $(LIB_SRC:.cpp=.o)
PROJ_DEP = ${APP_OBJ:.o=.d} ${LIB_OBJ:.o=.d}

LIB_VER = ${PROJ_MAJORVER}.${PROJ_MINORVER}

# TODO: fix up for empty library
ifeq (,${LIB_SRC})
  LIB_NAMES =
else
  LIB_NAMES = lib${PROJ_NAME}.a lib${PROJ_NAME}.so.${LIB_VER}

  ifeq ($(DRWN_SHARED_LIBS), 0)
    PROJLIB = $(BIN_PATH)/lib${PROJ_NAME}.a
  else
    PROJLIB = $(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER}
    LFLAGS += -l${PROJ_NAME}
  endif
endif

.PHONY: clean lib
.PRECIOUS: $(APP_OBJ) $(LIB_OBJ)

ifeq ($(wildcard $(EXT_PATH)/opencv), $(EXT_PATH)/opencv)
  all: lib ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
else
  all: warning
endif
lib: ${addprefix ${BIN_PATH}/,$(LIB_NAMES)}

# applications
$(BIN_PATH)/%: %.o $(PROJLIB) $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(PROJLIB) $(LFLAGS)

# static libraries
$(BIN_PATH)/lib${PROJ_NAME}.a: $(LIB_OBJ)
	ar rcs $@ $(LIB_OBJ)

# shared libraries
$(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER}: $(LIB_OBJ)
	${CCC} ${CFLAGS} -shared -Wl,-soname,lib${PROJ_NAME}.so.${PROJ_MAJORVER} -o $(BIN_PATH)/lib${PROJ_NAME}.so.${LIB_VER} $(LIB_OBJ)
	${LDCONFIG} -n $(BIN_PATH)

# darwin libraries
$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

warning:
	@echo "** PROJECT REQUIRES OPENCV TO BE INSTALLED **"
	false

# dependencies and object files
%.o : %.cpp
	$(CCC) -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
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


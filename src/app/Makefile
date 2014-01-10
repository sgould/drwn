# DARWIN APPLICATIONS MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..

-include $(DRWN_PATH)/make.mk

#######################################################################
# add application source files here
#######################################################################
APP_SRC = persistentStorageManager.cpp \
	evalClassifier.cpp learnClassifier.cpp

APP_SRC += testDarwinBase.cpp testDarwinIO.cpp testDarwinML.cpp \
	testDarwinPGM.cpp testDarwinClassifiers.cpp

ifeq ($(DRWN_BUILD_VISION_LIB), 1)
  APP_SRC += generateSuperpixels.cpp grabCut.cpp highOrderGrabCut.cpp \
	testDarwinVision.cpp viewObjectDetections.cpp visualizeImageDataset.cpp \
	visualizeSuperpixels.cpp
endif

#######################################################################

APP_PROG_NAMES = $(APP_SRC:.cpp=)
APP_OBJ = $(APP_SRC:.cpp=.o)

.PHONY: clean
.PRECIOUS: $(APP_OBJ)

all: depend ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}

$(BIN_PATH)/%: %.o $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(LFLAGS)

.cpp.o:
	${CCC} ${CFLAGS} -c $< -o $@

$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

depend:
	g++ ${CFLAGS} -MM ${APP_SRC} >depend

clean:
	-rm $(APP_OBJ) *.o *.d
	-rm ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
	-rm depend

-include depend

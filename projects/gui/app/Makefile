# DARWIN COMMANDLINE APPLICATIONS MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..
PROJ_PATH := $(shell pwd)/..

-include $(DRWN_PATH)/make.mk
-include $(PROJ_PATH)/make.mk

#######################################################################
# add application source files here
#######################################################################
APP_SRC = databaseUtility.cpp demoClassifier.cpp

OTHER_SRC =

#######################################################################

APP_PROG_NAMES = $(APP_SRC:.cpp=)
APP_OBJ = $(APP_SRC:.cpp=.o)
OTHER_OBJ = $(OTHER_SRC:.cpp=.o)

.PHONY: clean
.PRECIOUS: $(APP_OBJ) $(OTHER_OBJ)

all: depend ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}

$(BIN_PATH)/%: %.o $(OTHER_OBJ) $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(OTHER_OBJ) $(LFLAGS)

.cpp.o:
	${CCC} ${CFLAGS} -c $< -o $@

$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

depend:
	g++ ${CFLAGS} -MM ${APP_SRC} >depend

clean:
	-rm $(APP_OBJ)
	-rm $(OTHER_OBJ)
	-rm ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
	-rm depend

-include depend

# DARWIN PROJECT MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DRWN_PATH := $(shell pwd)/../..
PROJ_PATH := $(shell pwd)

-include $(DRWN_PATH)/make.mk

#######################################################################
# add application source files here
#######################################################################

APP_SRC = darwin.cpp

OTHER_SRC = drwnStatusBar.cpp drwnIconFactory.cpp drwnOptionsEditor.cpp \
	drwnMatrixEditor.cpp drwnTextEditor.cpp

#######################################################################

CFLAGS += -I$(PROJ_PATH)/include

APP_PROG_NAMES = $(APP_SRC:.cpp=)
APP_OBJ = $(APP_SRC:.cpp=.o)
OTHER_OBJ = $(OTHER_SRC:.cpp=.o)

.PHONY: clean external libs apps docs plugins
.PRECIOUS: $(APP_OBJ) $(OTHER_OBJ)

ifeq ($(wildcard $(EXT_PATH)/wx), $(EXT_PATH)/wx)
all: depend libs apps
else
all: warning
endif

apps: ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}

libs:
	cd $(PROJ_PATH)/lib && $(MAKE) && cd $(PROJ_PATH)

plugins:
	for i in $(shell ls plugins); do cd "plugins/$${i}" && ($(MAKE) || true) && cd ${PROJ_PATH}; done

$(BIN_PATH)/%: %.o $(OTHER_OBJ) $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(OTHER_OBJ) $(LFLAGS)

.cpp.o:
	${CCC} ${CFLAGS} -c $< -o $@

$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

warning:
	@echo "** GUI PROJECT REQUIRES WXWIDGETS TO BE INSTALLED **"
	false

depend:
	g++ ${CFLAGS} -MM ${APP_SRC} >depend

clean:
	-cd lib && $(MAKE) clean && cd ../..
	-rm $(APP_OBJ)
	-rm $(OTHER_OBJ)
	-rm ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
	for i in $(shell ls plugins); do cd "plugins/$${i}" && ($(MAKE) clean || true) && cd ${PROJ_PATH}; done
	-rm depend

-include depend

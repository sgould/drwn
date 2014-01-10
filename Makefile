# DARWIN LINUX LIBRARY AND APPLICATIONS MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

#######################################################################
# DO NOT EDIT THIS MAKEFILE UNLESS YOU KNOW WHAT YOU ARE DOING. EDIT
# THE APPROPRIATE MAKEFILE IN THE PROJECT OR SANDBOX DIRECTORY.
#######################################################################

ifeq ($(wildcard ./make.local), ./make.local)
  -include ./make.local
endif

DRWN_PATH = $(shell pwd)

.PHONY: clean external drwnlibs drwnapps drwnprojs drwndocs

all: depend external drwnlibs drwnapps

external:

drwnapps:
	cd src/app && $(MAKE) && cd ../..

drwnprojs:
	for i in $(shell ls projects); do cd "projects/$${i}" && ($(MAKE) || true) && cd ${DRWN_PATH}; done
	for i in ${DRWN_PROJECTS}; do cd "$${i}" && ($(MAKE) || true) && cd ${DRWN_PATH}; done

drwnlibs:
	cd src/lib && $(MAKE) && cd ../..

drwndocs:
	cd src/doc && $(MAKE) && cd ..

depend:
	cd src/app && $(MAKE) depend && cd ../..

clean:
	-cd src/lib && $(MAKE) clean && cd ../..
	-cd src/app && $(MAKE) clean && cd ../..
	-cd src/doc && $(MAKE) clean && cd ..
	for i in $(shell ls projects); do cd "projects/$${i}" && ($(MAKE) clean || true) && cd ${DRWN_PATH}; done
	-cd bin && rm -f * && cd ..
	find . -name "*~" -delete

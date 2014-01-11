# DARWIN DOCUMENTATION MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

all: html

html:
	doxygen doxygen.conf || true

.PHONY: clean html

clean:
	-rm *~
	-rm -r html

# DARWIN DOCUMENTATION MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

all: html pdf

html:
	doxygen doxygen.conf || true

pdf:
	pdflatex cheatsheet.tex || true

.PHONY: clean html pdf

clean:
	-rm *~
	-rm -r html
	-rm cheatsheet.aux
	-rm cheatsheet.log
	-rm cheatsheet.pdf

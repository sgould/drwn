# TUTORIAL MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DARWIN = "../../../"

INC_DIRS = -I${DARWIN}/include -I${DARWIN}/external
LIBS = -L${DARWIN}/bin -ldrwnML -ldrwnPGM -ldrwnIO -ldrwnBase -lm -lpthread

main: cmdline xmlio

cmdline:
	g++ -g -o cmdline cmdline.cpp ${INC_DIRS} ${LIBS}

xmlio:
	g++ -g -o xmlio xmlio.cpp ${INC_DIRS} ${LIBS}

clean:
	rm -f cmdline xmlio TheSimpsons.xml

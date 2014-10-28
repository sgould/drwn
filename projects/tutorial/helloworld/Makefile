# HELLOWORLD MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DARWIN = "../../../"

INC_DIRS = -I${DARWIN}/include -I${DARWIN}/external
LIBS = -L${DARWIN}/bin -ldrwnML -ldrwnPGM -ldrwnIO -ldrwnBase -lm -lpthread

ifeq ($(shell uname), Darwin)
    LIBS += -stdlib=libstdc++
endif

main:
	g++ -g -o helloworld helloworld.cpp ${INC_DIRS} ${LIBS}

clean:
	rm -f helloworld

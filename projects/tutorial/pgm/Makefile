# TUTORIAL MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DARWIN = "../../../"

INC_DIRS = -I${DARWIN}/include -I${DARWIN}/external
LIBS = -L${DARWIN}/bin -ldrwnML -ldrwnPGM -ldrwnIO -ldrwnBase -lm -lpthread

ifeq ($(shell uname), Darwin)
    LIBS += -stdlib=libstdc++
endif

main: viterbi factors crf3d

viterbi: viterbi.cpp
	g++ -g -o viterbi viterbi.cpp ${INC_DIRS} ${LIBS}

factors: factors.cpp
	g++ -g -o factors factors.cpp ${INC_DIRS} ${LIBS}

crf3d: crf3d.cpp
	g++ -g -o crf3d crf3d.cpp ${INC_DIRS} ${LIBS}

clean:
	rm -f viterbi factors crf3d

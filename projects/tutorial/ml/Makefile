# TUTORIAL MAKEFILE
# Stephen Gould <stephen.gould@anu.edu.au>

DARWIN = "../../../"

INC_DIRS = -I${DARWIN}/include -I${DARWIN}/external
LIBS = -L${DARWIN}/bin -ldrwnML -ldrwnPGM -ldrwnIO -ldrwnBase -lm -lpthread

ifeq ($(shell uname), Darwin)
    LIBS += -stdlib=libstdc++
endif

main: gaussian transform pca regression classify

gaussian: gaussian.cpp
	g++ -g -o gaussian gaussian.cpp ${INC_DIRS} ${LIBS}

transform: transform.cpp
	g++ -g -o transform transform.cpp ${INC_DIRS} ${LIBS}

pca: pca.cpp
	g++ -g -o pca pca.cpp ${INC_DIRS} ${LIBS}

regression: regression.cpp
	g++ -g -o regression regression.cpp ${INC_DIRS} ${LIBS}

classify: classify.cpp
	g++ -g -o classify classify.cpp ${INC_DIRS} ${LIBS}

clean:
	rm -f gaussian transform pca regression classify

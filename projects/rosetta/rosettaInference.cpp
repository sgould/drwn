/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    rosettaInference.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Example inference routines on Rosetta Protein Design dataset.
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnPGM.h"

using namespace std;

// function prototypes ------------------------------------------------------

void runInference(const char *name, drwnMAPInference *inf, double de);

// main ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./rosettaInference [OPTIONS] <graph>\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    // load graph
    DRWN_LOG_VERBOSE("loading " << DRWN_CMDLINE_ARGV[0] << "...");
    drwnFactorGraph graph;
    graph.read(DRWN_CMDLINE_ARGV[0]);

    // remove constant factors
    const double de = drwnFactorGraphUtils::removeUniformFactors(graph);

    // run iterated conditional modes inference
    {
        drwnICMInference icm(graph);
        runInference("icm", &icm, de);
    }

    // run asynchronous max-product inference
    {
        drwnAsyncMaxProdInference mp(graph);
        runInference("async-max-prod", &mp, de);
    }

    // run TRW-S inference
    {
        drwnTRWSInference trws(graph);
        runInference("trw-s", &trws, de);
    }

    // run gemplp inference
    {
        drwnGEMPLPInference gemplp(graph);
        runInference("gemplp", &gemplp, de);
    }

    // run Sontag et al., UAI 2008 inference
    {
        drwnSontag08Inference sontag(graph);
        runInference("sontag08", &sontag, de);
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// helper functions ---------------------------------------------------------

void runInference(const char *name, drwnMAPInference *inf, double de)
{
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle(name));
    drwnFullAssignment assignment;
    pair <double, double> e = inf->inference(assignment);
    DRWN_LOG_MESSAGE(name << " returned energy " << e.first + de);
    if (e.second > -DRWN_DBL_MAX) {
        DRWN_LOG_MESSAGE(name << " returned lower bound " << e.second + de);
    }
    DRWN_LOG_VERBOSE(name << " returned assignment " << toString(assignment));
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle(name));
}

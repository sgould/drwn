/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinPGM.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Regression tests for classes in the drwnPGM library.
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

// prototypes ---------------------------------------------------------------

template<class T>
void testMaxFlow();
void testMaxFlowOnFile(const char *filename);
void testTableFactorMapping();
void testFactors();
void testFactorStorage();
void testFactorOperations();
void testFactorGraph();
void testFactorGraphMemory();
void testSumProduct(const char *filename);
void testAsyncSumProduct(const char *filename);
void testFullInference(const char *filename);
void testMAPInference(const char *name, const char *filename);
void testSpeed(int speedTestLoops, int speedTestCard);

// main ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinPGM [OPTIONS] (<test>)*\n";
    cerr << "OPTIONS:\n"
         << "  -f <filename>     :: filename for subsequent test\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "TESTS:\n"
         << "  maxflow           :: maxflow/mincut tests\n"
         << "  mapping           :: table factor mapping tests\n"
         << "  factor            :: basic factor tests\n"
         << "  operations        :: test factor operations\n"
         << "  storage           :: factor storage tests\n"
         << "  graph             :: factor graph tests\n"
         << "  memory            :: factor graph memory tests\n"
         << "  sumprod <fname>   :: test sum-product inference\n"
         << "  asumprod <fname>  :: test asynchronous sum-product inference\n"
         << "  fullinf <fname>   :: test full inference\n"
         << "  map <inf> <fname> :: test MAP inference (<inf> can be one of\n";

    list<string> mapOptions = drwn::breakString(toString(drwnMAPInferenceFactory::get().getRegisteredClasses()), 56);
    for (list<string>::const_iterator it = mapOptions.begin(); it != mapOptions.end(); ++it) {
        cerr << "                       " << *it << "\n";
    }

    cerr << "  speed <n> <m>     :: speed test loops (n) and factor size (m)\n"
	 << endl;
}

int main(int argc, char *argv[])
{
    const char *filename = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-f", filename)
        DRWN_CMDLINE_FLAG_BEGIN("maxflow")
            if (filename == NULL) {
                testMaxFlow<drwnBKMaxFlow>();
                testMaxFlow<drwnEdmondsKarpMaxFlow>();
            } else {
                testMaxFlowOnFile(filename);
                filename = NULL;
            }
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("mapping")
            testTableFactorMapping();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("factor")
            testFactors();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("storage")
            testFactorStorage();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("operations")
            testFactorOperations();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("graph")
            testFactorGraph();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("memory")
            testFactorGraphMemory();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("fullinf", p)
            testFullInference(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("sumprod", p)
            testSumProduct(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("asumprod", p)
            testAsyncSumProduct(p[0]);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("map", p)
            testMAPInference(p[0], p[1]);
        DRWN_CMDLINE_OPTION_END(2)
        DRWN_CMDLINE_OPTION_BEGIN("speed", p)
            testSpeed(atoi(p[0]), atoi(p[1]));
        DRWN_CMDLINE_OPTION_END(2)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// tests ------------------------------------------------------------------

template<class T>
void testMaxFlow()
{
    {
        // mincut is 7
        T graph;
        graph.addNodes(2);
        graph.addSourceEdge(0, 4);
        graph.addSourceEdge(1, 3);
        graph.addEdge(0, 1, 3);
        graph.addTargetEdge(0, 4);
        graph.addTargetEdge(1, 5);

        DRWN_LOG_MESSAGE("maxflow: " << graph.solve());
    }

    {
        // mincut is 250
        T graph;
        graph.addNodes(6);
        graph.addSourceEdge(0, 100);
        graph.addSourceEdge(1, 200);
        graph.addSourceEdge(2, 150);
        graph.addEdge(0, 3, DRWN_DBL_MAX);
        graph.addEdge(0, 4, DRWN_DBL_MAX);
        graph.addEdge(1, 4, DRWN_DBL_MAX);
        graph.addEdge(2, 5, DRWN_DBL_MAX);
        graph.addTargetEdge(3, 200);
        graph.addTargetEdge(4, 100);
        graph.addTargetEdge(5, 50);

        DRWN_LOG_MESSAGE("maxflow: " << graph.solve());
    }

    {
        // mincut is 5
        T graph;
        graph.addNodes(5);
        graph.addSourceEdge(0, 3);
        graph.addSourceEdge(2, 3);
        graph.addEdge(0, 1, 4);
        graph.addEdge(1, 2, 1);
        graph.addEdge(1, 3, 2);
        graph.addEdge(2, 3, 2);
        graph.addEdge(2, 4, 6);
        graph.addTargetEdge(3, 1);
        graph.addTargetEdge(4, 9);

        DRWN_LOG_MESSAGE("maxflow: " << graph.solve());

        vector<int> cut(graph.numNodes());
        for (unsigned i = 0; i < cut.size(); i++) {
            cut[i] = graph.inSetS(i) ? 0 : 1;
        }
        DRWN_LOG_MESSAGE("    cut:" << toString(cut));
    }

    {
        // dynamic graphcuts; mincut is 7
        T graph;
        graph.addNodes(2);
        graph.addSourceEdge(0, 4);
        graph.addSourceEdge(1, 3);
        graph.addEdge(0, 1, 3);
        graph.addTargetEdge(0, 1);
        graph.addTargetEdge(1, 6);

        DRWN_LOG_MESSAGE("dyanmic maxflow: " << graph.solve());
        graph.addEdge(0, 1, -2);
        graph.solve();

        graph.addEdge(0, 1, 2);
        DRWN_LOG_MESSAGE("dynamic maxflow: " << graph.solve());
    }

    {
        // dynamic graphcuts; mincut is 5
        T graph;
        graph.addNodes(5);
        graph.addSourceEdge(0, 3);
        graph.addSourceEdge(2, 3);
        graph.addEdge(0, 1, 4);
        graph.addEdge(1, 2, 1);
        graph.addEdge(1, 3, 2);
        graph.addEdge(2, 3, 2);
        graph.addEdge(2, 4, 6);
        graph.addTargetEdge(3, 1);
        graph.addTargetEdge(4, 9);
        DRWN_LOG_MESSAGE("dynamic maxflow 2: " << graph.solve());

        graph.addEdge(0, 1, -4);
        graph.addEdge(1, 2, 1);
        graph.addEdge(1, 3, -2);
        graph.addEdge(2, 3, 2);
        graph.addEdge(2, 4, -6);
        DRWN_LOG_MESSAGE("dynamic maxflow 2: " << graph.solve());

        graph.addEdge(0, 1, 4);
        graph.addEdge(1, 2, -1);
        graph.addEdge(1, 3, 2);
        graph.addEdge(2, 3, -2);
        graph.addEdge(2, 4, 6);
        DRWN_LOG_MESSAGE("dynamic maxflow 2: " << graph.solve());
    }
}

void testMaxFlowOnFile(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_FCN_TIC;

    const int MAX_NODES = DRWN_INT_MAX; //50000;
    DRWN_LOG_VERBOSE("constructing graph from " << filename << "...");
#if 1
    drwnBKMaxFlow graph;
#else
    drwnEdmondsKarpMaxFlow graph;
#endif
    ifstream ifs(filename);

    int u, v;
    double cap;

    while (!ifs.eof()) {
        ifs >> u >> v >> cap;
        if (ifs.fail()) break;

        if ((u >= MAX_NODES) || (v >= MAX_NODES)) {
            DRWN_LOG_WARNING_ONCE("truncating graph to " << MAX_NODES << " nodes");
            continue;
        }

        if ((u >= (int)graph.numNodes()) || (v >= (int)graph.numNodes())) {
            graph.addNodes(std::max(u, v) - graph.numNodes() + 1);
        }
        if (u < 0) {
            graph.addSourceEdge(v, cap);
        } else if (v < 0) {
            graph.addTargetEdge(u, cap);
        } else {
            graph.addEdge(u, v, cap);
        }
    }

    ifs.close();
    DRWN_LOG_VERBOSE("...graph has " << graph.numNodes() << " nodes");

    DRWN_LOG_MESSAGE("maxflow: " << graph.solve());
    DRWN_FCN_TOC;
}


// Tests the mapping between two factors.
void testTableFactorMapping()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(2, "b");
    universe->addVariable(2, "c");
    universe->addVariable(2, "d");

    int dstVars[] = {0, 1, 2, 3};
    int srcVars[] = {2, 0};

    drwnTableFactorMapping m(vector<int>(&dstVars[0], &dstVars[3] + 1),
        vector<int>(&srcVars[0], &srcVars[1] + 1), universe);

    DRWN_LOG_VERBOSE("begin() is " << *m.begin());
    DRWN_LOG_VERBOSE("begin()[5] is " << m.begin()[5]);
    DRWN_LOG_VERBOSE("end() is " << *m.end());

    unsigned dstIndx = 0;
    drwnTableFactorMapping::iterator it = m.begin();
    while (it != m.end()) {
        cout << dstIndx << "\t" << *it << "\n";
        ++it, ++dstIndx;
    }
    cout << "---\n";
    while (it != m.begin()) {
        --it, --dstIndx;
        cout << dstIndx << "\t" << *it << "\n";
    }
}

// These tests are from Daphne Koller and Nir Friedman's text book,
// "Structured Probabilistic Models," MIT Press, 2009.
void testFactors()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(2, "b");
    universe->addVariable(2, "c");
    universe->addVariable(2, "d");

    universe->print();

    // define factors
    drwnTableFactor psiA(universe);
    drwnTableFactor psiB(universe);

    psiA.addVariable("b");
    psiA.addVariable("a");
    psiB.addVariable("c");
    psiB.addVariable("b");

    // populate directly, but could also use use:
    //  psiA[psiA.indexOf(0, <a>, psiA.indexOf(1, <b>))] = <x>;
    psiA[0] = 0.5; psiA[1] = 0.8; psiA[2] = 0.1;
    psiA[3] = 0.0; psiA[4] = 0.3; psiA[5] = 0.9;
    psiA.dump();

    psiB[0] = 0.5; psiB[1] = 0.7;
    psiB[2] = 0.1; psiB[3] = 0.2;
    psiB.dump();

    // multiply factors together
    drwnTableFactor psiC(universe);
    //drwnFactorProductOp opCEqualsATimesB(&psiC, &psiA, &psiB);
    //opCEqualsATimesB.execute();
    drwnFactorProductOp(&psiC, &psiA, &psiB).execute();
    psiC.dump();

    // marginalize "b"
    drwnTableFactor psiD(universe);
    //drwnFactorMarginalizeOp opDEqualsCMarginalized(&psiD, &psiC, 1);
    //opDEqualsCMarginalized.execute();
    drwnFactorMarginalizeOp(&psiD, &psiC, universe->findVariable("b")).execute();
    psiD.dump();

#if 1
    // check results
    const double EXPECTED_RESULTS[3][2] = {
        {0.33, 0.51}, {0.05, 0.07}, {0.24, 0.39}
    };
    for (int a = 0; a < 3; a++) {
        for (int c = 0; c < 2; c++) {
            if (fabs(psiD[psiD.indexOf(0, a, psiD.indexOf(2, c))] - EXPECTED_RESULTS[a][c]) > 1.0e-16) {
                cerr << "ERROR: term mismatch " << psiD[psiD.indexOf(0, a, psiD.indexOf(2, c))]
                    << " != " << EXPECTED_RESULTS[a][c] << endl;
		assert(false);
            }
        }
    }
#endif
}

void testFactorStorage()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(2, "b");
    universe->addVariable(2, "c");
    universe->addVariable(2, "d");

    universe->print();

    // define factors using shared storage
    drwnTableFactorStorage storage;
    drwnTableFactor psiA(universe, &storage);
    drwnTableFactor psiB(universe, &storage);

    psiA.addVariable("b");
    psiA.addVariable("a");
    psiB.addVariable("c");
    psiB.addVariable("b");

    // populate
    psiA[0] = 0.5; psiA[1] = 0.8; psiA[2] = 0.1;
    psiA[3] = 0.0; psiA[4] = 0.3; psiA[5] = 0.9;
    psiA.dump();

    psiB[0] = 0.5; psiB[1] = 0.7;
    psiB[2] = 0.1; psiB[3] = 0.2;
    psiB.dump();
    psiA.dump();
}

void testFactorOperations()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(2, "b");
    universe->addVariable(2, "c");
    universe->addVariable(2, "d");

    drwnTableFactor psiA(universe);
    drwnTableFactor psiB(universe);

    psiA.addVariable("a");
    psiA.addVariable("b");
    psiA.addVariable("c");
    psiA.addVariable("d");

    drwnInitializeRand();
    for (unsigned i = 0; i < psiA.entries(); i++) {
        psiA[i] = (double)(rand() % 100);
    }
    psiA.dump();

    drwnFactorReduceOp(&psiB, &psiA, 1, 0).execute();
    psiB.dump();

    drwnFactorReduceOp(&psiB, &psiA, 1, 1).execute();
    psiB.dump();
}

void testFactorGraph()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(2, "b");
    universe->addVariable(2, "c");
    universe->addVariable(2, "d");

    // create factor graph
    drwnFactorGraph graph(universe);

    // add first factor
    drwnTableFactor psi(universe);
    psi.addVariable("b");
    psi.addVariable("a");

    psi[0] = 0.5; psi[1] = 0.8; psi[2] = 0.1;
    psi[3] = 0.0; psi[4] = 0.3; psi[5] = 0.9;

    graph.copyFactor(&psi);

    // add second factor
    psi = drwnTableFactor(universe);
    psi.addVariable("c");
    psi.addVariable("b");

    psi[0] = 0.5; psi[1] = 0.7;
    psi[2] = 0.1; psi[3] = 0.2;

    graph.copyFactor(&psi);

    // connect the graph
    graph.connectGraph();

    // dump
    graph.dump();

    // run MAP inference
    drwnICMInference icm(graph);
    drwnFullAssignment assignment;
    double e = icm.inference(assignment).first;
    DRWN_LOG_MESSAGE("map assignment has energy " << e);
    DRWN_LOG_VERBOSE("map assignment is " << toString(assignment));
}

void testFactorGraphMemory()
{
    int grid = 25;
    while (1) {
        DRWN_LOG_MESSAGE("creating factor graph on " << grid << "-by-" << grid << " grid...");

        // define variables on a grid
        drwnVarUniversePtr universe(new drwnVarUniverse(grid * grid, 2));

        // create factor graph
        drwnFactorGraph graph(universe);

        // add random unary factors
        for (int i = 0; i < grid * grid; i++) {
            drwnTableFactor *psi = new drwnTableFactor(universe);
            psi->addVariable(i);
            graph.addFactor(psi);
        }

        // add random pairwise factors
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid - 1; j++) {
                drwnTableFactor *psi = new drwnTableFactor(universe);
                psi->addVariable(i * grid + j);
                psi->addVariable(i * grid + j + 1);
                graph.addFactor(psi);

                psi = new drwnTableFactor(universe);
                psi->addVariable(j * grid + i);
                psi->addVariable((j + 1) * grid + i);
                graph.addFactor(psi);
            }
        }

        DRWN_LOG_MESSAGE("...graph has " << graph.numFactors() << " factors");

        // increase size of grid
        grid += 25;
    }
}

void testFullInference(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_FCN_TIC;

    // load graph
    drwnFactorGraph graph;
    graph.read(filename);

    // build full energy table
    drwnVarUniversePtr universe(graph.getUniverse());
    drwnTableFactor factor(universe);
    for (int i = 0; i < universe->numVariables(); i++) {
        factor.addVariable(i);
    }

    for (int i = 0; i < graph.numFactors(); i++) {
        drwnFactorAdditionOp op(&factor, &factor, dynamic_cast<drwnTableFactor *>(graph[i]));
        op.execute();
    }

    drwnFullAssignment assignment;
    factor.assignmentOf(factor.indexOfMin(), assignment);
    double e = graph.getEnergy(assignment);
    DRWN_LOG_MESSAGE("map assignment has energy " << e);
    DRWN_LOG_VERBOSE("map assignment is " << toString(assignment));
    DRWN_FCN_TOC;
}

void testSumProduct(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_FCN_TIC;

    // load graph
    drwnFactorGraph graph;
    graph.read(filename);

#if 0
    for (int i = 0; i < graph.numFactors(); i++) {
        for (size_t j = 0; j < graph[i]->entries(); j++) {
            (*graph[i])[j] = exp(-1.0 * (*graph[i])[j]);
        }
    }
#endif

    // run inference
    drwnSumProdInference inf(graph);
    inf.inference();

    for (int i = 0; i < graph.getUniverse()->numVariables(); i++) {
        inf[i].dump();
    }

    DRWN_FCN_TOC;
}

void testAsyncSumProduct(const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_FCN_TIC;

    // load graph
    drwnFactorGraph graph;
    graph.read(filename);

    // run inference
    drwnAsyncSumProdInference inf(graph);
    inf.inference();

    for (int i = 0; i < graph.numFactors(); i++) {
        inf.marginal(*graph[i]);
    }
    graph.dump();

    DRWN_FCN_TOC;
}

void testMAPInference(const char *name, const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    DRWN_FCN_TIC;

    // load graph
    drwnFactorGraph graph;
    graph.read(filename);
    if (graph.numEdges() == 0) {
        graph.connectGraph();
    }

    double de = drwnFactorGraphUtils::removeUniformFactors(graph);

    // run MAP inference
    drwnMAPInference *inf = drwnMAPInferenceFactory::get().create(name, graph);
    DRWN_ASSERT(inf != NULL);
    drwnFullAssignment assignment;
    pair<double, double> e = inf->inference(assignment);
    DRWN_LOG_MESSAGE("map assignment has energy " << e.first + de);
    DRWN_LOG_MESSAGE("lower bound energy is " << e.second + de);
    DRWN_LOG_VERBOSE("map assignment is " << toString(assignment));
    delete inf;

    DRWN_FCN_TOC;
}

void testSpeed(int speedTestLoops, int speedTestCard)
{
    int h = drwnCodeProfiler::getHandle("drwnFactor speed test");

    // define variables
    DRWN_LOG_VERBOSE("defining universe...");
    drwnVarUniversePtr universe(new drwnVarUniverse(3, speedTestCard));
    universe->dump();

    // create factors
    DRWN_LOG_VERBOSE("creating factors...");
    drwnTableFactor tmpA(universe), tmpB(universe),
        tmpC(universe), tmpD(universe);
    tmpA.addVariable(0);
    tmpA.addVariable(1);
    tmpB.addVariable(1);
    tmpB.addVariable(2);

    for (size_t i = 0; i < tmpA.entries(); i++) {
        tmpA[i] = drand48();
    }

    for (size_t i = 0; i < tmpB.entries(); i++) {
        tmpB[i] = drand48();
    }

    // create factor operations
    DRWN_LOG_VERBOSE("creating factor operations...");
    drwnFactorProductOp tmpOp1(&tmpC, &tmpA, &tmpB);
    drwnFactorMarginalizeOp tmpOp2(&tmpD, &tmpC, 1);

    DRWN_LOG_VERBOSE("running test...");
    for (int i = 0; i < speedTestLoops; i++) {
	drwnCodeProfiler::tic(h);
	tmpOp1.execute();
	tmpOp2.execute();
	drwnCodeProfiler::toc(h);
    }
}

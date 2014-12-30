/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    crf3d.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Example code for constructing a Markov random field over a 3d grid. Uses
**  shared memory for the Potts model.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnPGM.h"

using namespace std;

// constants -----------------------------------------------------------------

const int Nx = 10;         //!< dimensions in the x direction
const int Ny = 10;         //!< dimensions in the y direction
const int Nz = 10;         //!< dimensions in the z direction
const int K = 2;           //!< number of classes
const double LAMBDA = 0.1; //!< strength of Potts potential

// utility routines ----------------------------------------------------------

//! converts from an (x,y,z) grid coordinate to a variable index
inline int node2indx(int x, int y, int z)
{
    return x + Nx * y + Nx * Ny * z;
}

//! converts from a variable index to an (x,y,z) grid coordinate
inline void indx2node(int indx, int& x, int& y, int& z)
{
    x = indx % Nx;
    y = (indx / Nx) % Ny;
    z = indx / (Nx * Ny);
}

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./crf3d [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // define universe
    drwnVarUniversePtr universe(new drwnVarUniverse(Nx * Ny * Nz, K));

    // initialize the factor graph
    drwnTableFactorStorage sharedPottsStorage(K * K);
    drwnFactorGraph graph(universe);

    // add random unary potentials for each grid location (x, y, z)
    for (int z = 0; z < Nz; z++) {
        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {

                // create the factor
                const int varIndx = node2indx(x, y, z);
                drwnTableFactor *phi = new drwnTableFactor(universe);
                phi->addVariable(varIndx);
                for (unsigned i = 0; i < phi->entries(); i++) {
                    (*phi)[i] = drand48();
                }

                // add it to the graph
                graph.addFactor(phi);
            }
        }
    }
                            
    // add potts potentials over 6-neighbourhood using shared storage
    for (int z = 0; z < Nz; z++) {
        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                
                // z neighbours
                if (z > 0) {
                    drwnTableFactor *phi = new drwnTableFactor(universe, &sharedPottsStorage);
                    phi->addVariable(node2indx(x, y, z));
                    phi->addVariable(node2indx(x, y, z - 1));
                    graph.addFactor(phi);
                }

                // y neighbours
                if (y > 0) {
                    drwnTableFactor *phi = new drwnTableFactor(universe, &sharedPottsStorage);
                    phi->addVariable(node2indx(x, y, z));
                    phi->addVariable(node2indx(x, y - 1, z));
                    graph.addFactor(phi);
                }

                // x neighbours
                if (x > 0) {
                    drwnTableFactor *phi = new drwnTableFactor(universe, &sharedPottsStorage);
                    phi->addVariable(node2indx(x, y, z));
                    phi->addVariable(node2indx(x - 1, y, z));
                    graph.addFactor(phi);
                }
            }
        }
    }

    // initialize values of potts potential
    sharedPottsStorage.fill(LAMBDA);
    for (int i = 0; i < K; i++) {
        sharedPottsStorage[i + K * i] = 0.0; // set diagonal to zero
    }

    DRWN_LOG_VERBOSE("Graph has " << graph.numFactors() << " factors");

    // run inference
    drwnAlphaExpansionInference infObj(graph);
    drwnFullAssignment mapAssignment;
    pair<double, double> energy = infObj.inference(mapAssignment);

    DRWN_LOG_VERBOSE(toString(mapAssignment));
    DRWN_LOG_MESSAGE("Min. Energy: " << energy.first);

    return 0;
}

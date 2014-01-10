/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    viterbi.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
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

// viterbi decoder -----------------------------------------------------------

drwnFullAssignment viterbi(vector<drwnTableFactor>& factors)
{
    DRWN_ASSERT(!factors.empty());
    const int T = (int)factors.size() + 1;
    drwnVarUniversePtr universe(factors[0].getUniverse());

    // check input assumptions
    for (int t = 0; t < T - 1; t++) {
        DRWN_ASSERT(factors[t].size() == 2);      // factor is pairwise
        DRWN_ASSERT(factors[t].hasVariable(t));   // factor has variable x_t
        DRWN_ASSERT(factors[t].hasVariable(t+1)); // factor has variable x_{t+1}
    }

    // forward pass (modifies factors)
    vector<drwnTableFactor> messages(T - 1, drwnTableFactor(universe));

    drwnFactorMaximizeOp(&messages[0], &factors[0], 0).execute();
    drwnFactorNormalizeOp(&messages[0]).execute();

    for (int t = 1; t < T - 1; t++) {
        drwnFactorTimesEqualsOp(&factors[t], &messages[t - 1]).execute();
        drwnFactorMaximizeOp(&messages[t], &factors[t], t).execute();
        drwnFactorNormalizeOp(&messages[t]).execute();
    }

    // backward pass
    drwnFullAssignment mapAssignment(T);
    mapAssignment[T - 1] = messages[T - 2].indexOfMax();
    for (int t = T - 2; t >= 0; t--) {
        drwnTableFactor mu(universe);
        drwnFactorReduceOp(&mu, &factors[t], t + 1, mapAssignment[t + 1]).execute();
        mapAssignment[t] = mu.indexOfMax();
    }

    return mapAssignment;
}

// main ----------------------------------------------------------------------

int main()
{
    // define universe
    const int T = 5; // number of time steps
    const int N = 3; // number of states per variable
    drwnVarUniversePtr universe(new drwnVarUniverse(T, N));

    // create markov chain
    vector<drwnTableFactor> factors(T - 1, drwnTableFactor(universe));
    for (int t = 0; t < T - 1; t++) {
        factors[t].addVariable(t);
        factors[t].addVariable(t + 1);
        for (int i = 0; i < N * N; i++) {
            factors[t][i] = drand48();
        }
        factors[t].normalize();
        factors[t].dump();
    }

    // run viterbi
    drwnFullAssignment assignment = viterbi(factors);
    DRWN_LOG_MESSAGE("x^\\star = " << toString(assignment));

    return 0;
}

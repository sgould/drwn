/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVarAssignment.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVarAssignment.h"

using namespace std;

void successor(drwnFullAssignment& assignment, const drwnVarUniverse& universe)
{
    for (unsigned i = 0; i < assignment.size(); i++) {
        assignment[i] += 1;
        if (assignment[i] >= universe.varCardinality(i)) {
            assignment[i] = 0;
        } else {
            break;
        }
    }
}

void predecessor(drwnFullAssignment& assignment, const drwnVarUniverse& universe)
{
    for (unsigned i = 0; i < assignment.size(); i++) {
        assignment[i] -= 1;
        if (assignment[i] < 0) {
            assignment[i] = universe.varCardinality(i) - 1;
        } else {
            break;
        }
    }
}

void successor(drwnPartialAssignment& assignment, const drwnVarUniverse& universe)
{
    for (drwnPartialAssignment::iterator it = assignment.begin(); it != assignment.end(); ++it) {
        it->second += 1;
        if (it->second >= universe.varCardinality(it->first)) {
            it->second = 0;
        } else {
            break;
        }
    }
}

void predecessor(drwnPartialAssignment& assignment, const drwnVarUniverse& universe)
{
    for (drwnPartialAssignment::iterator it = assignment.begin(); it != assignment.end(); ++it) {
        it->second -= 1;
        if (it->second < 0) {
            it->second = universe.varCardinality(it->first) - 1;
        } else {
            break;
        }
    }
}

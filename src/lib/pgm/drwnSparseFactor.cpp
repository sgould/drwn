/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSparseFactor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Albert Chen <chenay@student.unimelb.edu.au>
**
*****************************************************************************/

#include <iostream>
#include "drwnSparseFactor.h"

using namespace std;
using namespace drwn;

void drwnSparseFactor::dfaToVals(const drwnFullAssignment& y,
    vector<int>& vals) const {
    DRWN_ASSERT(vals.empty());
    const vector<int>& vars = getOrderedVars();

    for (vector<int>::const_iterator vi = vars.begin(); vi != vars.end(); ++vi) {
        DRWN_ASSERT(y[*vi] != -1);
        vals.push_back(y[*vi]);
    }
}

double drwnSparseFactor::getValueOf(const drwnFullAssignment& y) const
{
    vector<int> vals;
    dfaToVals(y, vals);

    if (assignments.find(vals) == assignments.end()) {
        return 0;
    }

    return assignments.at(vals);
}

double drwnSparseFactor::getValueOf(const drwnPartialAssignment& y) const {
    vector<int> vals;

    for (vector<int>::const_iterator vi = _variables.begin(); vi != _variables.end(); ++vi) {
        vals.push_back(y.at(*vi));
    }

    if (assignments.find(vals) == assignments.end()) {
        return 0;
    }

    return assignments.at(vals);
}

void drwnSparseFactor::setValueOf(const drwnFullAssignment& y, double val)
{
    vector<int> vals;
    dfaToVals(y, vals);
    assignments[vals] = val;
}

void drwnSparseFactor::setValueOf(const drwnPartialAssignment& y, double val)
{
    vector<int> vals;

    for (vector<int>::const_iterator vi = _variables.begin(); vi != _variables.end(); ++vi) {
        vals.push_back(y.at(*vi));
    }

    assignments[vals] = val;
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
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

double drwnSparseFactor::getValueOf(const drwnFullAssignment& y) const
{
    drwnLocalAssignment vals;
    dfaToVals(y, vals);
    
    map<drwnLocalAssignment, double>::const_iterator it = _assignments.find(vals);
    return (it == _assignments.end()) ? 0 : it->second;
}

double drwnSparseFactor::getValueOf(const drwnPartialAssignment& y) const 
{
    drwnLocalAssignment vals;
    for (vector<int>::const_iterator vi = _variables.begin(); vi != _variables.end(); ++vi) {
        vals.push_back(y.at(*vi));
    }

    map<drwnLocalAssignment, double>::const_iterator it = _assignments.find(vals);
    return (it == _assignments.end()) ? 0 : it->second;
}

void drwnSparseFactor::setValueOf(const drwnFullAssignment& y, double val)
{
    vector<int> vals;
    dfaToVals(y, vals);
    _assignments[vals] = val;
}

void drwnSparseFactor::setValueOf(const drwnPartialAssignment& y, double val)
{
    vector<int> vals;

    for (vector<int>::const_iterator vi = _variables.begin(); vi != _variables.end(); ++vi) {
        vals.push_back(y.at(*vi));
    }

    _assignments[vals] = val;
}

void drwnSparseFactor::dfaToVals(const drwnFullAssignment& y, drwnLocalAssignment& vals) const 
{
    DRWN_ASSERT(vals.empty());
    const vector<int>& vars = getOrderedVars();

    for (drwnLocalAssignment::const_iterator vi = vars.begin(); vi != vars.end(); ++vi) {
        DRWN_ASSERT(y[*vi] != -1);
        vals.push_back(y[*vi]);
    }
}

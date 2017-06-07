/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRegression.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <cassert>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "drwnBase.h"
#include "drwnDataset.h"
#include "drwnRegression.h"

using namespace std;

// drwnRegression -----------------------------------------------------------

drwnRegression::drwnRegression() : drwnProperties(),
    _nFeatures(0), _bValid(false)
{
    // do nothing
}

drwnRegression::drwnRegression(unsigned n) : drwnProperties(),
    _nFeatures((int)n), _bValid(false)
{
    // do nothing
}

drwnRegression::drwnRegression(const drwnRegression &r) : drwnProperties(),
    _nFeatures(r._nFeatures), _bValid(r._bValid)
{
    // do nothing
}

// initialization
void drwnRegression::initialize(unsigned n)
{
    _nFeatures = (int)n;
    _bValid = false;
}

// i/o
bool drwnRegression::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "nFeatures", toString(_nFeatures).c_str(), false);
    return true;
}

bool drwnRegression::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "nFeatures") != NULL);
    initialize(atoi(drwnGetXMLAttribute(xml, "nFeatures")));

    _bValid = true;
    return _bValid;
}

// training
double drwnRegression::train(const vector<vector<double> >& features,
    const vector<double>& targets)
{
    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()),
        "size mismatch between features and targets");

    drwnRegressionDataset dataset;
    dataset.features = features;
    dataset.targets = targets;

    return train(dataset);
}

double drwnRegression::train(const vector<vector<double> >& features,
    const vector<double>& targets, const vector<double>& weights)
{
    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()) && (features.size() == weights.size()),
        "size mismatch between features, targets, and weights");

    drwnRegressionDataset dataset;
    dataset.features = features;
    dataset.targets = targets;
    dataset.weights = weights;

    return train(dataset);
}

double drwnRegression::train(const char *filename)
{
    drwnRegressionDataset dataset(filename);
    return train(dataset);
}

// evaluation (regression)
void drwnRegression::getRegressions(const vector<vector<double> >& features,
    vector<double>& outputTargets) const
{
    DRWN_ASSERT(_bValid);
    outputTargets.resize(features.size());
    for (unsigned i = 0; i < features.size(); i++) {
        outputTargets[i] = getRegression(features[i]);
    }
}

/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnClassifier.cpp
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
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;

// drwnClassifier -----------------------------------------------------------

drwnClassifier::drwnClassifier() : drwnProperties(),
    _nFeatures(0), _nClasses(0), _bValid(false)
{
    // do nothing
}

drwnClassifier::drwnClassifier(unsigned n, unsigned k) : drwnProperties(),
    _nFeatures((int)n), _nClasses((int)k), _bValid(false)
{
    // do nothing
}

drwnClassifier::drwnClassifier(const drwnClassifier &c) : drwnProperties(),
    _nFeatures(c._nFeatures), _nClasses(c._nClasses), _bValid(c._bValid)
{
    // do nothing
}

// initialization
void drwnClassifier::initialize(unsigned n, unsigned k)
{
    _nFeatures = (int)n;
    _nClasses = (int)k;
    _bValid = false;
}

// i/o
bool drwnClassifier::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "nFeatures", toString(_nFeatures).c_str(), false);
    drwnAddXMLAttribute(xml, "nClasses", toString(_nClasses).c_str(), false);

    // write standard (non-read-only) options
    writeProperties(xml);

    return true;
}

bool drwnClassifier::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(!drwnIsXMLEmpty(xml));

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "nFeatures") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "nClasses") != NULL);

    initialize(atoi(drwnGetXMLAttribute(xml, "nFeatures")),
        atoi(drwnGetXMLAttribute(xml, "nClasses")));

    // read standard (non-read-only) options
    readProperties(xml);

    _bValid = true;
    return _bValid;
}

// training
double drwnClassifier::train(const vector<vector<double> >& features,
    const vector<int>& targets)
{
    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()),
        "size mismatch between features and labels");

    drwnClassifierDataset dataset;
    dataset.features = features;
    dataset.targets = targets;

    return train(dataset);
}

double drwnClassifier::train(const vector<vector<double> >& features,
    const vector<int>& targets, const vector<double>& weights)
{
    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()) && (features.size() == weights.size()),
        "size mismatch between features, labels, and weights");

    drwnClassifierDataset dataset;
    dataset.features = features;
    dataset.targets = targets;
    dataset.weights = weights;

    return train(dataset);
}

double drwnClassifier::train(const char *filename)
{
    drwnClassifierDataset dataset(filename);
    return train(dataset);
}

// evaluation (log-probability)
void drwnClassifier::getClassScores(const vector<vector<double> >& features,
    vector<vector<double> >& outputScores) const
{
    DRWN_ASSERT(_bValid);
    outputScores.resize(features.size());
    for (unsigned i = 0; i < features.size(); i++) {
        getClassScores(features[i], outputScores[i]);
    }
}

// evaluation (normalized marginals)
void drwnClassifier::getClassMarginals(const vector<double>& features,
    vector<double>& outputMarginals) const
{
    DRWN_ASSERT(_bValid);
    getClassScores(features, outputMarginals);
    drwn::expAndNormalize(outputMarginals);
}

void drwnClassifier::getClassMarginals(const vector<vector<double> >& features,
    vector<vector<double> >& outputMarginals) const
{
    DRWN_ASSERT(_bValid);
    outputMarginals.resize(features.size());
    for (unsigned i = 0; i < features.size(); i++) {
        getClassMarginals(features[i], outputMarginals[i]);
    }
}

// evaluation (classification)
int drwnClassifier::getClassification(const vector<double>& features) const
{
    DRWN_ASSERT(_bValid);
    vector<double> scores;
    getClassScores(features, scores);
    return drwn::argmax(scores);
}

void drwnClassifier::getClassifications(const vector<vector<double> >& features,
    vector<int>& outputLabels) const
{
    DRWN_ASSERT(_bValid);
    outputLabels.resize(features.size());
    vector<double> scores;
    for (unsigned i = 0; i < features.size(); i++) {
        getClassScores(features[i], scores);
        outputLabels[i] = drwn::argmax(scores);
    }
}

// drwnClassifierFactory ----------------------------------------------------

void drwnFactoryTraits<drwnClassifier>::staticRegistration()
{
    // register known classifiers
    DRWN_FACTORY_REGISTER(drwnClassifier, drwnBoostedClassifier);
    DRWN_FACTORY_REGISTER(drwnClassifier, drwnCompositeClassifier);
    DRWN_FACTORY_REGISTER(drwnClassifier, drwnDecisionTree);
    DRWN_FACTORY_REGISTER(drwnClassifier, drwnMultiClassLogistic);
    DRWN_FACTORY_REGISTER(drwnClassifier, drwnRandomForest);
}


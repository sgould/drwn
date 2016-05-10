/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCrossValidator.cpp
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
#include "drwnConfusionMatrix.h"
#include "drwnCrossValidator.h"

using namespace std;

// drwnCrossValidator -------------------------------------------------------

// i/o

bool drwnCrossValidator::save(drwnXMLNode& xml) const
{
    for (map<string, list<string> >::const_iterator it = _settings.begin();
         it != _settings.end(); ++it) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "property", NULL, false);
        drwnAddXMLAttribute(*node, "name", it->first.c_str(), false);
        drwnAddXMLText(*node, toString(it->second).c_str());
    }
    return true;
}

bool drwnCrossValidator::load(drwnXMLNode& xml)
{
    DRWN_TODO;
    return false;
}

// number of trials
unsigned drwnCrossValidator::trials() const
{
    if (_settings.empty())
        return 0;

    unsigned n = 1;
    for (map<string, list<string> >::const_iterator it = _settings.begin();
         it != _settings.end(); ++it) {
        n *= it->second.size();
    }

    return n;
}

// clear settings
void drwnCrossValidator::clear(const char *property)
{
    DRWN_ASSERT(property != NULL);

    for (map<string, list<string> >::iterator it = _settings.begin();
         it != _settings.end(); ++it) {
        if (it->first == string(property)) {
            _settings.erase(it);
        }
    }
}

// add parameter settings
void drwnCrossValidator::add(const char *property, const char *value)
{
    DRWN_ASSERT((property != NULL) && (value != NULL));

    map<string, list<string> >::iterator it = _settings.find(string(property));
    if (it == _settings.end()) {
        it = _settings.insert(_settings.end(), make_pair(string(property), list<string>()));
    }

    it->second.push_back(string(value));
}

void drwnCrossValidator::addLinear(const char *property, double startValue,
    double endValue, int numSteps)
{
    DRWN_ASSERT(property != NULL);
    DRWN_ASSERT((startValue < endValue) && (numSteps > 1));

    map<string, list<string> >::iterator it = _settings.find(string(property));
    if (it == _settings.end()) {
        it = _settings.insert(_settings.end(), make_pair(string(property), list<string>()));
    }

    const double delta = (endValue - startValue) / (double)(numSteps - 1);
    for (int i = 0; i < numSteps - 1; i++) {
        it->second.push_back(toString(startValue));
        startValue += delta;
    }
    it->second.push_back(toString(endValue));
}

void drwnCrossValidator::addLogarithmic(const char *property, double startValue,
    double endValue, int numSteps)
{
    DRWN_ASSERT(property != NULL);
    DRWN_ASSERT((startValue > 0.0) && (startValue < endValue) && (numSteps > 1));

    map<string, list<string> >::iterator it = _settings.find(string(property));
    if (it == _settings.end()) {
        it = _settings.insert(_settings.end(), make_pair(string(property), list<string>()));
    }

    startValue = log(startValue);
    const double delta = (log(endValue) - startValue) / (double)(numSteps - 1);
    for (int i = 0; i < numSteps - 1; i++) {
        it->second.push_back(toString(exp(startValue)));
        startValue += delta;
    }
    it->second.push_back(toString(endValue));
}

double drwnCrossValidator::crossValidate(drwnClassifier * &classifier,
    const drwnClassifierDataset& trainSet, const drwnClassifierDataset& testSet) const
{
    DRWN_ASSERT(classifier != NULL);

    // initialize trials
    unsigned numTrials = 1;
    vector<list<string>::const_iterator> indexes;
    indexes.reserve(_settings.size());
    for (map<string, list<string> >::const_iterator it = _settings.begin();
         it != _settings.end(); ++it) {
        indexes.push_back(it->second.begin());
        numTrials *= it->second.size();
    }

    // iterate through trials
    unsigned trialNumber = 1;
    double bestScore = -DRWN_DBL_MAX;
    while (1) {
        DRWN_LOG_VERBOSE("cross-validating trial " << trialNumber << " of " << numTrials << "...");

        // train classifier
        drwnClassifier *c = classifier->clone();

        // set parameters
        vector<list<string>::const_iterator>::iterator jt = indexes.begin();
        map<string, list<string> >::const_iterator it;
        for (it = _settings.begin(); it != _settings.end(); ++it, ++jt) {
            c->setProperty(it->first.c_str(), (*jt)->c_str());
        }

        // train classifier
        c->train(trainSet);

        const double s = score(c, testSet);
        DRWN_LOG_VERBOSE("...with score " << s);
        if (s > bestScore) {
            bestScore = s;
            std::swap(c, classifier);
        }
        delete c;

        // increment to next setting
        trialNumber += 1;
        it = _settings.begin();
        jt = indexes.begin();
        while (it != _settings.end()) {
            if (++(*jt) == it->second.end()) {
                *jt = it->second.begin();
            } else {
                break;
            }
            ++it; ++jt;
        }

        if (it == _settings.end()) break;
    }

    return bestScore;
}

double drwnCrossValidator::score(drwnClassifier const *classifier,
    const drwnClassifierDataset& testSet) const
{
    DRWN_ASSERT(classifier != NULL);

    drwnConfusionMatrix confusion(classifier->numClasses());
    confusion.accumulate(testSet, classifier);
    return confusion.accuracy();
}


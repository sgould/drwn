/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnClassificationResults.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <limits>

#include "drwnBase.h"
#include "drwnClassificationResults.h"

using namespace std;

// drwnClassificationResults static data members ----------------------------

bool drwnClassificationResults::INCLUDE_MISSES = false;

// drwnClassificationResults ------------------------------------------------

drwnClassificationResults::drwnClassificationResults() :
    _numPositiveSamples(0), _numNegativeSamples(0), _posWeight(1.0)
{
    // do nothing
}

drwnClassificationResults::drwnClassificationResults(const drwnClassificationResults& c) :
    _scoredResults(c._scoredResults),
    _numPositiveSamples(c._numPositiveSamples),
    _numNegativeSamples(c._numNegativeSamples),
    _posWeight(c._posWeight)
{
    // do nothing
}

drwnClassificationResults::~drwnClassificationResults()
{
    // do nothing
}

// i/o
void drwnClassificationResults::clear()
{
    _scoredResults.clear();
    _numPositiveSamples = 0;
    _numNegativeSamples = 0;
    _posWeight = 1.0;
}

bool drwnClassificationResults::write(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);

    ofstream ofs(filename);
    DRWN_ASSERT(!ofs.fail());

    ofs << _scoredResults.size() << " "
        << _numPositiveSamples << " "
        << _numNegativeSamples << " "
        << _posWeight << "\n";
    for (map<double, pair<int, int> >::const_iterator p = _scoredResults.begin();
         p != _scoredResults.end(); p++) {
        ofs << p->first << " " << p->second.first << " " << p->second.second << "\n";
    }
    ofs.close();
    return true;
}

bool drwnClassificationResults::read(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    clear();
    ifstream ifs(filename);
    DRWN_ASSERT(!ifs.fail());

    int numPoints;
    ifs >> numPoints >> _numPositiveSamples >> _numNegativeSamples >> _posWeight;
    while (numPoints-- > 0) {
        double threshold;
        int posCount, negCount;

        ifs >> threshold >> posCount >> negCount;
        DRWN_ASSERT(!ifs.fail() && (posCount >= 0) && (negCount >= 0));
        _scoredResults.insert(make_pair(threshold, make_pair(posCount, negCount)));
    }

    ifs.close();
    return true;
}

// add points to the curve
void drwnClassificationResults::accumulate(const drwnClassificationResults& c)
{
    for (map<double, pair<int, int> >::const_iterator p = c._scoredResults.begin();
         p != c._scoredResults.end(); p++) {
        map<double, pair<int, int> >::iterator it = _scoredResults.find(p->first);
        if (it == _scoredResults.end()) {
            _scoredResults.insert(*p);
        } else {
            it->second.first += p->second.first;
            it->second.second += p->second.second;
        }
    }

    _numPositiveSamples += c._numPositiveSamples;
    _numNegativeSamples += c._numNegativeSamples;
}

void drwnClassificationResults::accumulate(const drwnClassifierDataset& dataset,
    drwnClassifier const *classifier, int positiveClassId)
{
    DRWN_ASSERT_MSG((classifier != NULL) && (positiveClassId >= 0) &&
        (positiveClassId < classifier->numClasses()), "id " << positiveClassId 
        << " invalid for " << classifier->numClasses() << "-class classifier");

    vector<double> marginals(classifier->numClasses());
    for (int i = 0; i < dataset.size(); i++) {
        if (dataset.targets[i] < 0) continue;
        classifier->getClassMarginals(dataset.features[i], marginals);
        if (dataset.targets[i] == 0) {
            accumulateNegatives(marginals[positiveClassId]);
        } else {
            accumulatePositives(marginals[positiveClassId]);
        }
    }
}

void drwnClassificationResults::accumulatePositives(double score, int count)
{
    DRWN_ASSERT(count >= 0);
    _numPositiveSamples += count;

    map<double, pair<int, int> >::iterator it = _scoredResults.find(score);
    if (it == _scoredResults.end()) {
        _scoredResults.insert(make_pair(score, make_pair(count, 0)));
    } else {
        it->second.first += count;
    }
}

void drwnClassificationResults::accumulatePositives(const vector<double>& scores)
{
    for (vector<double>::const_iterator it = scores.begin(); it != scores.end(); it++) {
        accumulatePositives(*it, 1);
    }
}

void drwnClassificationResults::accumulateNegatives(double score, int count)
{
    DRWN_ASSERT(count >= 0);
    _numNegativeSamples += count;

    map<double, pair<int, int> >::iterator it = _scoredResults.find(score);
    if (it == _scoredResults.end()) {
        _scoredResults.insert(make_pair(score, make_pair(0, count)));
    } else {
        it->second.second += count;
    }
}

void drwnClassificationResults::accumulateNegatives(const vector<double>& scores)
{
    for (vector<double>::const_iterator it = scores.begin(); it != scores.end(); it++) {
        accumulateNegatives(*it, 1);
    }
}

void drwnClassificationResults::accumulateMisses(int count)
{
    DRWN_ASSERT(count >= 0);
    _numPositiveSamples += count;
}

// drwnPRCurve --------------------------------------------------------------

drwnPRCurve::drwnPRCurve() : drwnClassificationResults()
{
    // do nothing
}

drwnPRCurve::drwnPRCurve(const drwnClassificationResults& c) :
    drwnClassificationResults(c)
{
    // do nothing
}

drwnPRCurve::~drwnPRCurve()
{
    // do nothing
}

vector<pair<double, double> > drwnPRCurve::getCurve() const
{
    vector<pair<double, double> > curve;
    if (_numPositiveSamples == 0) {
        return curve;
    }

    int posSum = INCLUDE_MISSES ? numMisses() : 0;
    int negSum = 0;
    for (map<double, pair<int, int> >::const_reverse_iterator it = _scoredResults.rbegin();
         it != _scoredResults.rend(); it++) {
        posSum += it->second.first;
        negSum += it->second.second;
        double recall = (double)posSum / (double)_numPositiveSamples;
        double precision = _posWeight * posSum / (_posWeight * posSum + negSum);
	if (!::isnan(recall) && !::isnan(precision)) {
            curve.push_back(make_pair(recall, precision));
        }
    }

    return curve;
}

void drwnPRCurve::writeCurve(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);
    const vector<pair<double, double> > curve(this->getCurve());
    
    ofstream ofs(filename);
    DRWN_ASSERT_MSG(!ofs.fail(), filename);
    for (vector<pair<double, double> >::const_iterator it = curve.begin();
         it != curve.end(); it++) {
        ofs << it->first << " " << it->second << "\n";
    }
    ofs.close();
}

double drwnPRCurve::averagePrecision(unsigned numPoints) const
{
    DRWN_ASSERT(numPoints > 2);
    numPoints = std::max(numPoints, (unsigned)_scoredResults.size());

    double ap = 0.0;

    int posSum = _numPositiveSamples - (INCLUDE_MISSES ? 0 : numMisses());
    int negSum = _numNegativeSamples;
    map<double, pair<int, int> >::const_iterator it = _scoredResults.begin();
    double recall = (double)posSum / (double)_numPositiveSamples;
    double precision = 0.0; //_posWeight * posSum / (_posWeight * posSum + negSum);
    for (unsigned i = 0; i < numPoints; i++) {
        double t = (double)(numPoints - i - 1) / (double)(numPoints - 1);
        while ((it != _scoredResults.end()) && (recall >= t)) {
            posSum -= it->second.first;
            negSum -= it->second.second;
            recall = (double)posSum / (double)_numPositiveSamples;
            precision = std::max(precision, _posWeight * posSum / (_posWeight * posSum + negSum));
            it++;
        }
        DRWN_LOG_DEBUG("...average precision interpolating (" << recall << ", " << precision << ")");
        ap += precision;
    }

    return ap / (double)numPoints;
}


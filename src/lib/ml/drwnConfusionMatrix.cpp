/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConfusionMatrix.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "drwnBase.h"
#include "drwnConfusionMatrix.h"

using namespace std;

// static data members ------------------------------------------------------

string drwnConfusionMatrix::COL_SEP("\t");
string drwnConfusionMatrix::ROW_BEGIN("\t");
string drwnConfusionMatrix::ROW_END("");

// member functions ---------------------------------------------------------

drwnConfusionMatrix::drwnConfusionMatrix(int n)
{
    _matrix.resize(n);
    for (int i = 0; i < n; i++) {
	_matrix[i].resize(n, 0);
    }
}

drwnConfusionMatrix::drwnConfusionMatrix(int n, int m)
{
    _matrix.resize(n);
    for (int i = 0; i < n; i++) {
	_matrix[i].resize(m, 0);
    }
}

drwnConfusionMatrix::~drwnConfusionMatrix()
{
    // do nothing
}

int drwnConfusionMatrix::numRows() const
{
    return (int)_matrix.size();
}

int drwnConfusionMatrix::numCols() const
{
    if (_matrix.empty())
        return 0;
    return (int)_matrix[0].size();
}

void drwnConfusionMatrix::clear()
{
    for (int i = 0; i < (int)_matrix.size(); i++) {
	fill(_matrix[i].begin(), _matrix[i].end(), 0);
    }
}

void drwnConfusionMatrix::accumulate(int actual, int predicted)
{
    if ((actual < 0) || (predicted < 0))
        return;
    _matrix[actual][predicted] += 1;
}

void drwnConfusionMatrix::accumulate(const drwnClassifierDataset& dataset,
    drwnClassifier const *classifier)
{
    DRWN_ASSERT(classifier != NULL);
    for (int i = 0; i < dataset.size(); i++) {
        if (dataset.targets[i] < 0) continue;
        int predicted = classifier->getClassification(dataset.features[i]);
        _matrix[dataset.targets[i]][predicted] += 1;
    }
}

void drwnConfusionMatrix::accumulate(const vector<int>& actual,
    const vector<int>& predicted)
{
    DRWN_ASSERT(actual.size() == predicted.size());
    for (unsigned i = 0; i < actual.size(); i++) {
	// treat < 0 as unknown
	if ((actual[i] < 0) || (predicted[i] < 0))
	    continue;
	DRWN_ASSERT(actual[i] < (int)_matrix.size());
	DRWN_ASSERT(predicted[i] < (int)_matrix[actual[i]].size());

	_matrix[actual[i]][predicted[i]] += 1;
    }
}

void drwnConfusionMatrix::accumulate(const drwnConfusionMatrix& confusion)
{
    DRWN_ASSERT(confusion._matrix.size() == _matrix.size());
    if (_matrix.empty()) return;

    DRWN_ASSERT(confusion._matrix[0].size() == _matrix[0].size());
    for (int a = 0; a < (int)_matrix.size(); a++) {
        for (int p = 0; p < (int)_matrix[a].size(); p++) {
            _matrix[a][p] += confusion._matrix[a][p];
        }
    }
}

void drwnConfusionMatrix::printCounts(ostream &os, const char *header) const
{
    if (header == NULL) {
        os << "--- confusion matrix: (actual, predicted) ---" << endl;
    } else {
        os << header << endl;
    }
    for (int i = 0; i < (int)_matrix.size(); i++) {
        os << ROW_BEGIN;
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
            if (j > 0) os << COL_SEP;
	    os << _matrix[i][j];
	}
	os << ROW_END << "\n";
    }
    os << "\n";
}

void drwnConfusionMatrix::printRowNormalized(ostream &os, const char *header) const
{
    if (header == NULL) {
        os << "--- confusion matrix: (actual, predicted) ---" << endl;
    } else {
        os << header << endl;
    }
    for (int i = 0; i < (int)_matrix.size(); i++) {
	double total = rowSum(i);
        os << ROW_BEGIN;
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
            if (j > 0) os << COL_SEP;
	    os << fixed << setprecision(3) << setw(4)
		 << ((double)_matrix[i][j] / total);
	}
	os << ROW_END << "\n";
    }
    os << "\n";
}

void drwnConfusionMatrix::printColNormalized(ostream &os, const char *header) const
{
    vector<double> totals;
    for (int i = 0; i < (int)_matrix[0].size(); i++) {
	totals.push_back(colSum(i));
    }

    if (header == NULL) {
        os << "--- confusion matrix: (actual, predicted) ---" << endl;
    } else {
        os << header << endl;
    }
    for (int i = 0; i < (int)_matrix.size(); i++) {
        os << ROW_BEGIN;
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
            if (j > 0) os << COL_SEP;
	    os << fixed << setprecision(3) << setw(4)
		 << ((double)_matrix[i][j] / totals[j]);
	}
	os << ROW_END << "\n";
    }
    os << "\n";
}

void drwnConfusionMatrix::printNormalized(ostream &os, const char *header) const
{
    double total = totalSum();

    if (header == NULL) {
        os << "--- confusion matrix: (actual, predicted) ---" << endl;
    } else {
        os << header << endl;
    }
    for (int i = 0; i < (int)_matrix.size(); i++) {
        os << ROW_BEGIN;
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
            if (j > 0) os << COL_SEP;
	    os << fixed << setprecision(3) << setw(4)
		 << ((double)_matrix[i][j] / total);
	}
	os << ROW_END << "\n";
    }
    os << "\n";
}

void drwnConfusionMatrix::printPrecisionRecall(ostream &os, const char *header) const
{
    if (header == NULL) {
        os << "--- class-specific recall/precision ---" << endl;
    } else {
        os << header << endl;
    }

    // recall
    os << ROW_BEGIN;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (i > 0) os << COL_SEP;
        double r = (_matrix[i].size() > i) ? (double)_matrix[i][i] / (double)rowSum(i) : 0.0;
        os << fixed << setprecision(3) << setw(4) << r;
    }
    os << ROW_END << "\n";

    // precision
    os << ROW_BEGIN;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (i > 0) os << COL_SEP;
        double p = (_matrix[i].size() > i) ? (double)_matrix[i][i] / (double)colSum(i) : 1.0;
        os << fixed << setprecision(3) << setw(4) << p;
    }
    os << ROW_END << "\n";
    os << "\n";
}

void drwnConfusionMatrix::printF1Score(ostream &os, const char *header) const
{
    if (header == NULL) {
        os << "--- class-specific F1 score ---" << endl;
    } else {
        os << header << endl;
    }

    os << ROW_BEGIN;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (i > 0) os << COL_SEP;
        // recall
        double r = (_matrix[i].size() > i) ? (double)_matrix[i][i] / (double)rowSum(i) : 0.0;
        // precision
        double p = (_matrix[i].size() > i) ? (double)_matrix[i][i] / (double)colSum(i) : 1.0;
        os << fixed << setprecision(3) << setw(4) << ((2.0 * p * r) / (p + r));
    }
    os << ROW_END << "\n";
    os << "\n";
}


void drwnConfusionMatrix::printJaccard(ostream &os, const char *header) const
{
    if (header == NULL) {
        os << "--- class-specific Jaccard coefficient ---" << endl;
    } else {
        os << header << endl;
    }

    os << ROW_BEGIN;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (i > 0) os << COL_SEP;
        double p = (_matrix[i].size() > i) ? (double)_matrix[i][i] /
            (double)(rowSum(i) + colSum(i) - _matrix[i][i]) : 0.0;
        os << fixed << setprecision(3) << setw(4) << p;
    }
    os << ROW_END << "\n";
    os << "\n";
}


void drwnConfusionMatrix::write(ostream &os) const
{
    printCounts(os);
}

void drwnConfusionMatrix::read(istream &is)
{
    for (int i = 0; i < (int)_matrix.size(); i++) {
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
	    is >> _matrix[i][j];
	}
    }
}

double drwnConfusionMatrix::rowSum(int n) const
{
    double v = 0.0;
    for (int i = 0; i < (int)_matrix[n].size(); i++) {
	v += (double)_matrix[n][i];
    }
    return v;
}

double drwnConfusionMatrix::colSum(int m) const
{
    double v = 0;
    for (int i = 0; i < (int)_matrix.size(); i++) {
	v += (double)_matrix[i][m];
    }
    return v;
}

double drwnConfusionMatrix::diagSum() const
{
    double v = 0;
    for (int i = 0; i < (int)_matrix.size(); i++) {
	if (i >= (int)_matrix[i].size())
	    break;
	v += (double)_matrix[i][i];
    }
    return v;
}

double drwnConfusionMatrix::totalSum() const
{
    double v = 0;
    for (int i = 0; i < (int)_matrix.size(); i++) {
	for (int j = 0; j < (int)_matrix[i].size(); j++) {
	    v += (double)_matrix[i][j];
	}
    }
    return v;
}

double drwnConfusionMatrix::accuracy() const
{
    return (diagSum() / totalSum());
}

double drwnConfusionMatrix::avgPrecision() const
{
    double totalPrecision = 0.0;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        totalPrecision += (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)colSum(i) : 1.0;
    }

    return totalPrecision /= (double)_matrix.size();
}

double drwnConfusionMatrix::avgRecall() const
{
    double totalRecall = 0.0;
    int numClasses = 0;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (_matrix[i].size() > i) {
            const double classSize = (double)rowSum(i);
            if (classSize > 0.0) {
                totalRecall += (double)_matrix[i][i] / classSize;
                numClasses += 1;
            }
        }
    }

    if (numClasses != (int)_matrix.size()) {
        DRWN_LOG_WARNING("not all classes represented in avgRecall()");
    }

    return totalRecall / (double)numClasses;
}

double drwnConfusionMatrix::avgJaccard() const
{
    double totalJaccard = 0.0;
    for (unsigned i = 0; i < _matrix.size(); i++) {
        if (_matrix[i].size() <= i) continue;
        const double intersectionSize = (double)_matrix[i][i];
        const double unionSize = (double)(rowSum(i) + colSum(i) - _matrix[i][i]);
        if (intersectionSize == unionSize) // avoid divide by zero
            totalJaccard += 1.0;
        else totalJaccard += intersectionSize / unionSize;
    }

    return totalJaccard / (double)_matrix.size();
}

double drwnConfusionMatrix::precision(int n) const
{
    DRWN_ASSERT(_matrix.size() > (unsigned)n);
    return (_matrix[n].size() > (unsigned)n) ?
        (double)_matrix[n][n] / (double)colSum(n) : 1.0;
}

double drwnConfusionMatrix::recall(int n) const
{
    DRWN_ASSERT(_matrix.size() > (unsigned)n);
    return (_matrix[n].size() > (unsigned)n) ?
        (double)_matrix[n][n] / (double)rowSum(n) : 0.0;
}

double drwnConfusionMatrix::jaccard(int n) const
{
    DRWN_ASSERT((_matrix.size() > (unsigned)n) && (_matrix[n].size() > (unsigned)n));
    const double intersectionSize = (double)_matrix[n][n];
    const double unionSize = (double)(rowSum(n) + colSum(n) - _matrix[n][n]);
    return (intersectionSize == unionSize) ? 1.0 :
        intersectionSize / unionSize;
}

const unsigned& drwnConfusionMatrix::operator()(int i, int j) const
{
    return _matrix[i][j];
}

unsigned& drwnConfusionMatrix::operator()(int i, int j)
{
    return _matrix[i][j];
}

// drwnConfusionMatrixConfig ------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnConfusionMatrix
//! \b colSep :: column separator (default: tab)\n
//! \b rowBegin :: start of row token (default: tab)\n
//! \b rowEnd :: end of row token (default: newline)\n

class drwnConfusionMatrixConfig : public drwnConfigurableModule {
public:
    drwnConfusionMatrixConfig() : drwnConfigurableModule("drwnConfusionMatrix") { }

    void usage(ostream &os) const {
        os << "      colSep        :: column separator (default: \""
           << drwnConfusionMatrix::COL_SEP << "\")\n"
           << "      rowBegin      :: start of row token (default: \""
           << drwnConfusionMatrix::ROW_BEGIN << "\")\n"
           << "      rowEnd        :: end of row token (default: \""
           << drwnConfusionMatrix::ROW_END << "\")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        // factor operation cache
        if (!strcmp(name, "colSep")) {
            drwnConfusionMatrix::COL_SEP = string(value);
        } else if (!strcmp(name, "rowBegin")) {
            drwnConfusionMatrix::ROW_BEGIN = string(value);
        } else if (!strcmp(name, "rowEnd")) {
            drwnConfusionMatrix::ROW_END = string(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnConfusionMatrixConfig gConfusionMatrixConfig;

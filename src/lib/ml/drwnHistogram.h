/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2013, Jason Corso, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnHistogram.h
** AUTHOR(S):   Jason Corso <jcorso@buffalo.edu>
**              Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**   Implements a simple templated histogram class.
**
*****************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

#include "drwnBase.h"

// drwnPCA class --------------------------------------------------------------
//!  Implements a simple templated histogram class.

template <typename TYPE>
class drwnHistogram {
 protected:
    double _mass;
    int _numBins;
    TYPE * _bins;
    TYPE _min;
    TYPE _max;

    // more transient values used in locating bins, etc...
    TYPE _alpha;
    TYPE _beta;

 public:
    //! constructor
    drwnHistogram(int n, double minVal, double maxVal) :
        _mass(0.0), _numBins(n), _bins(NULL), _min(minVal), _max(maxVal) {
        DRWN_ASSERT(minVal < maxVal);
        _bins = (TYPE *)malloc(sizeof(TYPE)*n);
        memset(_bins, 0, sizeof(TYPE)*n);
        _alpha = ((double)_numBins) / (_max - _min);
        _beta = 1.0 / _alpha;
    }

    //! copy constructor
    drwnHistogram(const drwnHistogram& h) :
        _mass(h._mass), _numBins(h._numBins), _bins(NULL), _min(h._min), _max(h._max),
        _alpha(h._alpha), _beta(h._beta) {
        _bins = (TYPE *)malloc(sizeof(TYPE) * _numBins);
        memcpy(_bins, h._bins,sizeof(TYPE) * _numBins);
    }

    //! destructor
    ~drwnHistogram() {
        if (_bins != NULL) {
            free(_bins);
            _bins = NULL;
        }
    }

    inline void addSample(TYPE s) {
        addWeightedSample(s, 1.0);
    }

    void addWeightedSample(TYPE s, double w) {
        const int binIndex = computeBin(s);

        if (_mass == 0.0) {
            _bins[binIndex] = 1;
            _mass = w;
        } else {
            const double oldmass = _mass;
            _mass += w;
            for (int i = 0; i < _numBins; i++) {
                if (i == binIndex) {
                    _bins[i] = (oldmass * _bins[i] + w) / _mass;
                } else {
                    _bins[i] = oldmass * _bins[i] / _mass;
                }
            }
        }
    }

    void appendToFile(FILE* fp) const {
        fprintf(fp,"%d\n", _numBins);
        for (int i = 0; i < _numBins; i++) {
            fprintf(fp, "%lf%s", (double)_bins[i], (i == _numBins - 1) ? "\n" : " ");
        }
    }

    //! compute and return the chi-squared distance between the two histograms
    //! d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2;
    //! If the xi+yi yields a bin with zero count, then I disregard this bin...
    double chiSquared(const drwnHistogram& H) const {
        DRWN_ASSERT(H._numBins == _numBins);
        double chi = 0.0;

        for (int i = 0; i < _numBins; i++) {
            const double a = _bins[i] + H._bins[i];
            if (a == 0.0) continue;

            double b = _bins[i] - H._bins[i];
            chi += b*b / a;
        }
        return 0.5 * chi;
    }

    void clear() {
        _mass = 0.0;
        memset(_bins, 0, sizeof(TYPE)*_numBins);
    }

    int computeBin(const TYPE& s) const {
        int binIndex;

        if (s <= _min) {
            binIndex = 0;
        } else if (s >= _max) {
            binIndex = _numBins - 1;
        } else {
            binIndex = std::max((int)(_alpha * (s - _min)), _numBins - 1);
        }

        return binIndex;
    }

    double entropy() const {
        double entropy = 0.0;
        for (int i = 0; i < _numBins; i++) {
            entropy -= (_bins[i] == 0) ? 0.0 : _bins[i] * log((double)_bins[i]);
        }
        return entropy;
    }

    void convertInternalToCDF() {
        for (int i = 1; i < _numBins; i++) {
            _bins[i] += _bins[i-1];
        }
    }

    double euclidean(const drwnHistogram& H) const {
        DRWN_ASSERT(H._numBins == _numBins);
        double euc = 0.0;

        for (int i = 0; i < _numBins; i++) {
            euc += (_bins[i] - H._bins[i]) * (_bins[i] - H._bins[i]);
        }

        return sqrt(euc);
    }

    double getBinCenter(int b) const { return _min + _beta*(double)b + (_beta / 2.0); }
    double getBinLeft(int b) const { return _min + _beta*(double)b; }
    double getBinRight(int b) const { return _min + _beta*(double)(b+1); }
    double getBinMass(int i) const { return _bins[i] * _mass; }
    double getBinWeight(int i) const { return _bins[i]; }

    double getBinWeightMax() const {
        double m = getBinWeight(0);
        for (int i = 1; i < _numBins; i++) {
            m = std::max(m, getBinWeight(i));
        }
        return m;
    }

    double getBinWeightMax(int &binIndex) const {
        double m = getBinWeight(0);
        binIndex = 0;
        for (int i = 1; i < _numBins; i++) {
            const double mm = getBinWeight(i);
            if (mm > m) {
                binIndex = i;
                m = mm;
            }
        }

        return m;
    }

    double getBinWeightSum() const {
        double sum = 0.0;
        for (int i = 0; i < _numBins; i++) {
            sum += getBinWeight(i);
        }
        return sum;
    }

    double getLikelihood(const TYPE& d) const { return _bins[computeBin(d)]; }

    double getMass() const { return _mass; }
    double getMax() const { return _max; }
    double getMin() const { return _min; }
    int getNumberOfBins() const { return _numBins; }

    double intersect(const drwnHistogram& H) const {
        DRWN_ASSERT(_numBins == H._numBins);
        double val = 0.0;

        for (int i = 0; i < _numBins; i++) {
            val += std::min(_bins[i], H._bins[i]);
        }

        return val;
    }

    //! compute the symmetric kl Difference (averaging the two assymetric)
    double klDistance(const drwnHistogram& H) const {
        return 0.5 * (klDivergence(H) + H.klDivergence(*this));
    }

    //! compute the one-way kldivergence from this to H
    double klDivergence(const drwnHistogram& H) const {
        double d = 0.0;
        for (int i = 0; i < _numBins; i++) {
            if ((_bins[i] != 0) && (H._bins[i] != 0)) {
                d += _bins[i] * log(_bins[i]/H._bins[i]);
            }
        }
        return d;
    }

    double l1distance(const drwnHistogram& H) const {
        double d = 0.0;
        for (int i = 0; i < _numBins; i++) {
            d += fabs(_bins[i] - H._bins[i]);
        }
        return d;
    }

    void mergeHistogram(const drwnHistogram& H) {
        for (int i = 0; i < _numBins; i++) {
            _bins[i] = (_bins[i] + H._bins[i]) / 2.0;
        }
        _mass += H.getMass();
    }

    //! only works for the case that the initial histogram here is 0 mass
    //! \todo needs to be extended
    void mergeWeightedHistogram(const drwnHistogram& H, double w) {
        for (int i = 0; i < _numBins; i++) {
            _bins[i] += H._bins[i] * w;
        }
        _mass += w;
    }

    void setAndNormalize(const drwnHistogram* H) {
        double invSum = 1.0 / H->getBinWeightSum();
        for (int i = 0; i < _numBins; i++) {
            _bins[i] = invSum * H->_bins[i];
        }
        _mass = H->_mass;
    }
};


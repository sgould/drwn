/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnClassificationResults.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>
#include <map>

#include "drwnBase.h"
#include "drwnDataset.h"
#include "drwnClassifier.h"

using namespace std;

// drwnClassificationResults ------------------------------------------------
//! Encapsulates summary of classifier output from which various curves can
//! be generated (e.g., precision-recall curves).
//!
//! The results can be scored arbitrarily so long as higher score implies
//! more likely. Positive and negative samples can be weighted differently.
//! The following code snippet shows how to produce a precision-recall curve
//! that can be subsequently plotted in Matlab.
//! \code
//!    vector<int> y = getGroundTruthLabels();
//!    vector<double> y_hat = getPredictionProbabilities();
//!
//!    drwnPRCurve curve;
//!    for (int n = 0; n < N; n++) {
//!        if (y[n] == 1) curve.accumulatePositives(y_hat[n]);
//!        else curve.accumulateNegatives(y_hat[n]);
//!    }
//!
//!    curve.writeCurve("pr.txt");
//! \endcode
//! 
//! \sa drwnPRCurve, \ref drwnTutorialML "drwnML Tutorial"

class drwnClassificationResults {
 public:
    static bool INCLUDE_MISSES; //!< true if some positive samples are never scored

 protected:
    //! number of positives (first) and negatives (second) grouped by score
    map<double, pair<int, int> > _scoredResults;
    int _numPositiveSamples;  //!< must be greater than sum(_scoredResults.first)
    int _numNegativeSamples;  //!< must be must be equal to sum(_scoredResults.second)
    double _posWeight;        //!< weight of positive-to-negative count

 public:
    //! default constructor
    drwnClassificationResults();
    //! copy constructor
    drwnClassificationResults(const drwnClassificationResults& c);
    virtual ~drwnClassificationResults();

    //! return the number os positive samples accumulated
    inline int numPositives() const { return _numPositiveSamples; }
    //! return the number of negative samples accumulated
    inline int numNegatives() const { return _numNegativeSamples; }
    //! return the total number (positive and negative) of samples accumulated
    inline int numSamples() const { return _numPositiveSamples + _numNegativeSamples; }
    //! return the number of unique classification scores
    inline int numThresholds() const { return (int)_scoredResults.size(); }
    //! return the number of positive samples that have not been scored
    inline int numMisses() const;

    //! return the relative weight of a positive sample to a negative sample
    inline double getPosWeight() const { return _posWeight; }
    //! set the relative weight of a positive sample to a negative sample
    inline void setPosWeight(double w) { DRWN_ASSERT(w > 0.0); _posWeight = w; }

    //! this will change the weight of the positive examples such that
    //! overall positive and negative examples will have the same weight
    inline void normalize();

    // i/o
    //! clear the accumulated scores
    void clear();
    //! write the accumulated scores to file
    bool write(const char *filename) const;
    //! read accumulated scores from file
    bool read(const char *filename);

    // modify the statistics
    //! accumulate results from another drwnClassificationResults object
    void accumulate(const drwnClassificationResults& c);
    //! Accumulate results from a classifier run on a dataset. The \p positiveClass
    //! parameter indicates the positive class label for multi-class classifiers.
    void accumulate(const drwnClassifierDataset& dataset,
        drwnClassifier const *classifier, int positiveClassId = 1);
    //! accumulate a single positive example
    void accumulatePositives(double score, int count = 1);
    //! accumulate multiple positive examples
    void accumulatePositives(const vector<double>& scores);
    //! accumulate a single negative example
    void accumulateNegatives(double score, int count = 1);
    //! accumulate multiple negative examples
    void accumulateNegatives(const vector<double> &scores);

    //! accumulate unscored positive examples (misses)
    void accumulateMisses(int count = 1);

    // reduce the number of points in the curve aggregating nearby scores
    // TODO: void quantize(double minThreshold = 0.0, double maxThreshold = 1.0, int numBins = 100);
};

// drwnPRCurve --------------------------------------------------------------
//! Precision-recall curve.
//!
//! \sa drwnClassificationResults, \ref drwnTutorialML "drwnML Tutorial"

class drwnPRCurve : public drwnClassificationResults {
 public:
    //! default constructor
    drwnPRCurve();
    //! copy constructor
    drwnPRCurve(const drwnClassificationResults& c);
    virtual ~drwnPRCurve();

    //! return a list of points defining the precision-recall curve
    vector<pair<double, double> > getCurve() const;
    //! write the precision-recall curve to a space-delimited file
    void writeCurve(const char *filename) const;
    //! extract the average precision (area under the curve)
    double averagePrecision(unsigned numPoints = 11) const;
};

// drwnClassificationResults inline functions -------------------------------

inline void drwnClassificationResults::normalize()
{
    if ((_numPositiveSamples > 0) && (_numNegativeSamples > 0)) {
        _posWeight = (double)_numNegativeSamples / (double)_numPositiveSamples;
    } else {
        _posWeight = 1.0;
    }
}

inline int drwnClassificationResults::numMisses() const {
    int count = _numPositiveSamples;
    for (map<double, pair<int, int> >::const_iterator it = _scoredResults.begin();
         it != _scoredResults.end(); it++) {
        count -= it->second.first;
    }
    return count;
}

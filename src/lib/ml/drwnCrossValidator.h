/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCrossValidator.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>
#include <map>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnDataset.h"
#include "drwnClassifier.h"

using namespace std;

// drwnCrossValidator -------------------------------------------------------
//! Utility class for cross-validating classifier meta-parameters by
//! brute-force testing of all combinations of some given settings.

class drwnCrossValidator : public drwnWriteable {
 protected:    
    map<string, list<string> > _settings;  //<! property/value list

 public:
    drwnCrossValidator() { /* do nothing */ }
    virtual ~drwnCrossValidator() { /* do nothing */ }

    // i/o
    const char *type() const { return "drwnCrossValidator"; }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    //! number of trials that will be run
    unsigned trials() const;

    //! clear all settings
    void clear() { _settings.clear(); }
    //! clear settings for a particular property
    void clear(const char *property);

    //! add a single parameter setting
    void add(const char *property, const char *value);
    //! add linear parameter range
    void addLinear(const char *property, double startValue, double endValue, int numSteps = 10);
    //! add logarithmic parameter range
    void addLogarithmic(const char *property, double startValue, double endValue, int numSteps = 10);

    //! run the cross-validation
    double crossValidate(drwnClassifier * &classifier,
        const drwnClassifierDataset& trainSet,
        const drwnClassifierDataset& testSet) const;

 protected:
    //! evaluation function (can be overridden in derived classes)
    virtual double score(drwnClassifier const *classifier,
        const drwnClassifierDataset& testSet) const;
};

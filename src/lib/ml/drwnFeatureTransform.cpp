/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnFeatureTransform.cpp
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

// drwnFeatureTransform -----------------------------------------------------

drwnFeatureTransform::drwnFeatureTransform() : drwnProperties(),
    _nFeatures(0), _bValid(false)
{
    // do nothing
}

drwnFeatureTransform::drwnFeatureTransform(const drwnFeatureTransform& t) :
    drwnProperties(), _nFeatures(t._nFeatures), _bValid(t._bValid)
{
    // do nothing
}

// i/o
bool drwnFeatureTransform::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "nFeatures", toString(_nFeatures).c_str(), false);
    return true;
}

bool drwnFeatureTransform::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "nFeatures"));
    _nFeatures = atoi(drwnGetXMLAttribute(xml, "nFeatures"));

    _bValid = true;
    return _bValid;
}

// evaluation
void drwnFeatureTransform::transform(vector<double>& x) const
{
    vector<double> y;
    transform(x, y);
    x.swap(y);
}

void drwnFeatureTransform::transform(vector<vector<double> >& x) const
{
    for (unsigned i = 0; i < x.size(); i++) {
        transform(x[i]);
    }
}

void drwnFeatureTransform::transform(const vector<vector<double> >& x,
    vector<vector<double> >& y) const
{
    y.resize(x.size());
    for (unsigned i = 0; i < x.size(); i++) {
        this->transform(x[i], y[i]);
    }
}

// drwnUnsupervisedTransform ------------------------------------------------

drwnUnsupervisedTransform::drwnUnsupervisedTransform() : drwnFeatureTransform()
{
    // do nothing
}

drwnUnsupervisedTransform::drwnUnsupervisedTransform(const drwnUnsupervisedTransform& t) :
    drwnFeatureTransform(t)
{
    // do nothing
}

// training
double drwnUnsupervisedTransform::train(const vector<vector<double> >& features,
    const vector<double>& weights)
{
    // default behaviour is to ignore weights
    DRWN_LOG_WARNING("ignoring weights for training " << this->type());
    return train(features);
}

// drwnSupervisedTransform --------------------------------------------------

drwnSupervisedTransform::drwnSupervisedTransform() : drwnFeatureTransform()
{
    // do nothing
}

drwnSupervisedTransform::drwnSupervisedTransform(const drwnSupervisedTransform& t) :
    drwnFeatureTransform(t)
{
    // do nothing
}

// training
double drwnSupervisedTransform::train(const vector<vector<double> >& features,
    const vector<int>& labels, const vector<double>& weights)
{
    // default behaviour is to ignore weights
    DRWN_LOG_WARNING("ignoring weights for training " << this->type());
    return train(features, labels);
}

// drwnFeatureTransformFactory ----------------------------------------------

void drwnFactoryTraits<drwnFeatureTransform>::staticRegistration()
{
    // register known classifiers
    DRWN_FACTORY_REGISTER(drwnFeatureTransform, drwnFeatureWhitener);
    DRWN_FACTORY_REGISTER(drwnFeatureTransform, drwnKMeans);
    DRWN_FACTORY_REGISTER(drwnFeatureTransform, drwnPCA);
    DRWN_FACTORY_REGISTER(drwnFeatureTransform, drwnFisherLDA);
    DRWN_FACTORY_REGISTER(drwnFeatureTransform, drwnLinearTransform);
}

/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiClassLogistic.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnMultiClassLogistic.h"

using namespace std;
using namespace Eigen;

// drwnMultiClassLogisticBase static members ------------------------------------

double drwnMultiClassLogisticBase::REG_STRENGTH = 1.0e-9;
int drwnMultiClassLogisticBase::MAX_ITERATIONS = 1000;

// drwnMultiClassLogisticBase ---------------------------------------------------

drwnMultiClassLogisticBase::drwnMultiClassLogisticBase() :
    drwnClassifier(), drwnOptimizer(), _regularizer(0), _lambda(REG_STRENGTH)
{
    // define properties
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnMultiClassLogisticBase::drwnMultiClassLogisticBase(unsigned n, unsigned k) :
    drwnClassifier(n, k), drwnOptimizer(), _regularizer(0), _lambda(REG_STRENGTH)
{
    // define properties
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));

    //initialize(n, k);
}

drwnMultiClassLogisticBase::drwnMultiClassLogisticBase(const drwnMultiClassLogisticBase &c) :
    drwnClassifier(c), drwnOptimizer(), _theta(c._theta),
    _regularizer(c._regularizer), _lambda(c._lambda)
{
    // define properties
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnMultiClassLogisticBase::~drwnMultiClassLogisticBase()
{
    // do nothing
}

// i/o
bool drwnMultiClassLogisticBase::save(drwnXMLNode& xml) const
{
    drwnClassifier::save(xml);
    drwnXMLUtils::serialize(xml, _theta);

    return true;
}

bool drwnMultiClassLogisticBase::load(drwnXMLNode& xml)
{
    // load classifier parameters
    drwnClassifier::load(xml);
    drwnXMLUtils::deserialize(xml, _theta);

    return true;
}

// training
double drwnMultiClassLogisticBase::train(const drwnClassifierDataset& dataset)
{
    if (dataset.hasWeights()) {
        return train(dataset.features, dataset.targets, dataset.weights);
    } else {
        return train(dataset.features, dataset.targets);
    }
}

double drwnMultiClassLogisticBase::train(const vector<vector<double> >& features,
    const vector<int>& targets)
{
    DRWN_FCN_TIC;

    // set pointer to data
    _features = &features;
    _targets = &targets;
    _weights = NULL;

    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()),
        "size mismatch between features and labels");

    drwnOptimizer::initialize(_theta.rows());
    Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows()) = _theta;

    double J = solve(MAX_ITERATIONS, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
    _theta = Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows());

    _features = NULL;
    _targets = NULL;
    _weights = NULL;

    _bValid = true;
    DRWN_FCN_TOC;
    return J;
}

double drwnMultiClassLogisticBase::train(const vector<vector<double> >& features,
    const vector<int>& targets, const vector<double>& weights)
{
    DRWN_FCN_TIC;

    // set pointer to data
    _features = &features;
    _targets = &targets;
    _weights = &weights;

    // check size
    DRWN_ASSERT_MSG((features.size() == targets.size()) && (features.size() == weights.size()), 
        "size mismatch between features, labels and weights");

    drwnOptimizer::initialize(_theta.rows());
    Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows()) = _theta;

    double J = solve(MAX_ITERATIONS, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
    _theta = Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows());

    _features = NULL;
    _targets = NULL;
    _weights = NULL;

    _bValid = true;
    DRWN_FCN_TOC;
    return J;
}

// drwnOptimizer interface
double drwnMultiClassLogisticBase::objective(const double *x) const
{
    double *df = new double[_n];
    double f = objectiveAndGradient(x, df);
    delete[] df;
    return f;
}

void drwnMultiClassLogisticBase::gradient(const double *x, double *df) const
{
    objectiveAndGradient(x, df);
}

// template instantiations --------------------------------------------------

#ifdef __LINUX__
//! \bug explicit template instantiation seems to cause compiler error C2908 in MSVC
template class drwnTMultiClassLogistic<drwnIdentityJointFeatureMap>;
template class drwnTMultiClassLogistic<drwnBiasJointFeatureMap>;
#endif

// drwnMultiClassLogisticConfig ---------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnMultiClassLogistic
//! \b lambda :: regularization strength (default: 1.0e-9)\n
//! \b maxIterations :: maximum number of training iterations (default: 1000)

class drwnMultiClassLogisticConfig : public drwnConfigurableModule {
public:
    drwnMultiClassLogisticConfig() : drwnConfigurableModule("drwnMultiClassLogistic") { }
    ~drwnMultiClassLogisticConfig() { }

    void usage(ostream &os) const {
        os << "      lambda        :: regularization strength (default: "
           << drwnMultiClassLogisticBase::REG_STRENGTH << ")\n";
        os << "      maxIterations :: maximum number of training iterations (default: "
           << drwnMultiClassLogisticBase::MAX_ITERATIONS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "lambda")) {
            drwnMultiClassLogisticBase::REG_STRENGTH = std::max(0.0, atof(value));
        } else if (!strcmp(name, "maxIterations")) {
            drwnMultiClassLogisticBase::MAX_ITERATIONS = std::max(0, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnMultiClassLogisticConfig gMultiClassLogisticConfig;

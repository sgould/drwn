/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearRegressor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnLinearRegressor.h"

using namespace std;
using namespace Eigen;

// drwnLinearRegressorBase static members -----------------------------------

double drwnLinearRegressorBase::HUBER_BETA = 1.0e-3;
double drwnLinearRegressorBase::REG_STRENGTH = 1.0e-9;
int drwnLinearRegressorBase::MAX_ITERATIONS = 1000;

// drwnLinearRegressorBase --------------------------------------------------

drwnLinearRegressorBase::drwnLinearRegressorBase() :
    drwnRegression(), drwnOptimizer(), _penalty(0), _beta(HUBER_BETA),
    _regularizer(0), _lambda(REG_STRENGTH)
{
    // define properties
    declareProperty("penalty", new drwnIntegerProperty(&_penalty));
    declareProperty("huberThreshold", new drwnDoubleProperty(&_beta));
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnLinearRegressorBase::drwnLinearRegressorBase(unsigned n) :
    drwnRegression(n), drwnOptimizer(), _penalty(0), _beta(HUBER_BETA),
    _regularizer(0), _lambda(REG_STRENGTH)
{
    // define properties
    declareProperty("penalty", new drwnIntegerProperty(&_penalty));
    declareProperty("huberThreshold", new drwnDoubleProperty(&_beta));
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));

    //initialize(n);
}

drwnLinearRegressorBase::drwnLinearRegressorBase(const drwnLinearRegressorBase &r) :
    drwnRegression(r), drwnOptimizer(), _theta(r._theta),
    _penalty(r._penalty), _beta(r._beta),
    _regularizer(r._regularizer), _lambda(r._lambda)
{
    // define properties
    declareProperty("penalty", new drwnIntegerProperty(&_penalty));
    declareProperty("huberThreshold", new drwnDoubleProperty(&_beta));
    declareProperty("regularizer", new drwnIntegerProperty(&_regularizer));
    declareProperty("regStrength", new drwnDoubleProperty(&_lambda));
}

drwnLinearRegressorBase::~drwnLinearRegressorBase()
{
    // do nothing
}

// i/o
bool drwnLinearRegressorBase::save(drwnXMLNode& xml) const
{
    drwnRegression::save(xml);
    drwnXMLUtils::serialize(xml, _theta);

    return true;
}

bool drwnLinearRegressorBase::load(drwnXMLNode& xml)
{
    drwnRegression::load(xml);
    drwnXMLUtils::deserialize(xml, _theta);

    return true;
}

// training
double drwnLinearRegressorBase::train(const drwnRegressionDataset& dataset)
{
    // set pointer to data
    _features = &dataset.features;
    _targets = &dataset.targets;
    _weights = dataset.hasWeights() ? &dataset.weights : NULL;

    drwnOptimizer::initialize(_theta.rows());
    Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows()) = _theta;

    double J = solve(MAX_ITERATIONS, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
    _theta = Eigen::Map<VectorXd>(drwnOptimizer::_x, _theta.rows());

    _features = NULL;
    _targets = NULL;
    _weights = NULL;

    _bValid = true;
    return J;
}

// drwnOptimizer interface
double drwnLinearRegressorBase::objective(const double *x) const
{
    double *df = new double[_n];
    double f = objectiveAndGradient(x, df);
    delete[] df;
    return f;
}

void drwnLinearRegressorBase::gradient(const double *x, double *df) const
{
    objectiveAndGradient(x, df);
}

// template instantiations --------------------------------------------------

#ifdef __LINUX__
//! \bug explicit template instantiation seems to cause compiler error C2908 in MSVC
template class drwnTLinearRegressor<drwnIdentityFeatureMap>;
template class drwnTLinearRegressor<drwnBiasFeatureMap>;
#endif

// drwnLinearRegressorConfig ------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnLinearRegressor
//! \b beta :: huber penalty threshold (default: 1.0e-3)\n
//! \b lambda :: regularization strength (default: 1.0e-9)\n
//! \b maxIterations :: maximum number of training iterations (default: 1000)

class drwnLinearRegressorConfig : public drwnConfigurableModule {
public:
    drwnLinearRegressorConfig() : drwnConfigurableModule("drwnLinearRegressor") { }
    ~drwnLinearRegressorConfig() { }

    void usage(ostream &os) const {
        os << "      beta          :: huber penalty threshold (default: "
           << drwnLinearRegressorBase::HUBER_BETA << ")\n";
        os << "      lambda        :: regularization strength (default: "
           << drwnLinearRegressorBase::REG_STRENGTH << ")\n";
        os << "      maxIterations :: maximum number of training iterations (default: "
           << drwnLinearRegressorBase::MAX_ITERATIONS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "beta")) {
            drwnLinearRegressorBase::HUBER_BETA = std::max(0.0, atof(value));
        } else if (!strcmp(name, "lambda")) {
            drwnLinearRegressorBase::REG_STRENGTH = std::max(0.0, atof(value));
        } else if (!strcmp(name, "maxIterations")) {
            drwnLinearRegressorBase::MAX_ITERATIONS = std::max(0, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnLinearRegressorConfig gLinearRegressorConfig;

/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGaussianMixture.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <cmath>

// drwn libraries
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnSuffStats.h"
#include "drwnGaussian.h"
#include "drwnGaussianMixture.h"

using namespace std;

// drwnGaussianMixtureThread ------------------------------------------------

class drwnGaussianMixtureThread : public drwnThreadJob {
protected:
    unsigned _n;
    const vector<vector<double> >& _x;
    const vector<drwnGaussian> *_g;
    const VectorXd *_logLambda;
    unsigned _offset;
    unsigned _step;

public:
    vector<drwnSuffStats> stats;
    double logLikelihood;

public:
    drwnGaussianMixtureThread(unsigned n, const vector<vector<double> >& x,
        const vector<drwnGaussian> *g, const VectorXd *logLambda, unsigned offset, unsigned step) : 
        _n(n), _x(x), _g(g), _logLambda(logLambda), _offset(offset), _step(step) {
        stats.resize(_g->size(), drwnSuffStats(_n, DRWN_PSS_FULL));
    }

    ~drwnGaussianMixtureThread() { /* do nothing */ }

    void operator()() {
        vector<double> y(_g->size());
        for (unsigned k = 0; k < _g->size(); k++) {
            stats[k].clear();
        }

        // E-step
        logLikelihood = 0.0;
        for (unsigned i = _offset; i < _x.size(); i += _step) {
            for (unsigned k = 0; k < _g->size(); k++) {
                y[k] = (*_logLambda)[k] + (*_g)[k].evaluateSingle(_x[i]);
            }
            const double logZ = drwn::expAndNormalize(y);

            // maintain sufficient statistics
            for (unsigned k = 0; k < _g->size(); k++) {
                stats[k].accumulate(_x[i], y[k]);
            }

            logLikelihood += logZ; // evaluateSingle(x[i]);
        }
    }
};

// drwnGaussianMixture ------------------------------------------------------

int drwnGaussianMixture::MAX_ITERATIONS = 100;

drwnGaussianMixture::drwnGaussianMixture(unsigned n, unsigned k) : _n(n)
{
    DRWN_ASSERT((n > 0) && (k > 0));
    initialize(n, k);
}

drwnGaussianMixture::~drwnGaussianMixture()
{
    // do nothing
}

// initialization
void drwnGaussianMixture::initialize(unsigned n, unsigned k)
{
    DRWN_ASSERT((n > 0) && (k > 0));

    // allocate means and covariances
    if ((_g.size() != k) || (_n != n)) {
        _n = n;
        _g.resize(k, drwnGaussian(_n));
        for (unsigned i = 0; i < k; i++) {
            _g[i] = drwnGaussian(VectorXd::Random(n), MatrixXd::Identity(n, n));
        }
    }

    // initialize mixture weights
    _logLambda = VectorXd::Constant(k, log(1.0 / (double)k));
}

// i/o
bool drwnGaussianMixture::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "n", toString(_n).c_str(), false);
    drwnAddXMLAttribute(xml, "k", toString(_g.size()).c_str(), false);

    drwnXMLNode *node = drwnAddXMLChildNode(xml, "lambda", NULL, false);
    drwnXMLUtils::serialize(*node, _logLambda);

    for (unsigned i = 0; i < _g.size(); i++) {
        node = drwnAddXMLChildNode(xml, _g[i].type());
        _g[i].save(*node);
    }

    return true;
}

bool drwnGaussianMixture::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "n") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "k") != NULL);

    initialize(atoi(drwnGetXMLAttribute(xml, "n")), atoi(drwnGetXMLAttribute(xml, "k")));

    // load mixture weights
    drwnXMLNode *node = xml.first_node("lambda");
    DRWN_ASSERT(node != NULL);
    drwnXMLUtils::deserialize(*node, _logLambda);
    DRWN_ASSERT(_logLambda.rows() == (int)_g.size());

    // load gaussian means and covariances
    DRWN_ASSERT(_g.empty() || (drwnCountXMLChildren(xml, _g[0].type()) == (int)_g.size()));
    for (unsigned i = 0; i < _g.size(); i++) {
        node = (i == 0) ? xml.first_node(_g[i].type()) : node->next_sibling(_g[i].type());
        _g[i].load(*node);
        DRWN_ASSERT(_g[i].dimension() == _n);
    }

    return true;
}

// evaluation
void drwnGaussianMixture::evaluate(const MatrixXd& x, VectorXd& p) const
{
    DRWN_ASSERT((x.cols() == (int)_n) && (x.rows() == p.rows()));
    for (int i = 0; i < x.rows(); i++) {
        p[i] = evaluateSingle(x.row(i));
    }
}

void drwnGaussianMixture::evaluate(const vector<vector<double> >& x, vector<double>& p) const
{
    DRWN_ASSERT(x.size() == p.size());
    for (unsigned i = 0; i < x.size(); i++) {
        p[i] = evaluateSingle(x[i]);
    }
}

double drwnGaussianMixture::evaluateSingle(const VectorXd& x) const
{
    double y = 0.0;
    for (unsigned k = 0; k < _g.size(); k++) {
        // TODO: numerical issues
        y += exp(_logLambda[k] + _g[k].evaluateSingle(x));
    }

    return log(y);
}

double drwnGaussianMixture::evaluateSingle(const vector<double>& x) const
{
    vector<double> y(_g.size());
    for (unsigned k = 0; k < _g.size(); k++) {
        y[k] += _logLambda[k] + _g[k].evaluateSingle(x);
    }
    return drwn::expAndNormalize(y);
}

void drwnGaussianMixture::sample(VectorXd &x) const
{
    DRWN_ASSERT(!_g.empty() && (x.rows() == (int)_n));

    // sample component
    int k = _logLambda.rows() - 1;
    double p = drand48() - exp(_logLambda[k]);
    while ((p > 0) && (k > 0)) {
        p -= exp(_logLambda[--k]);
    }

    // sample from gaussian
    _g[k].sample(x);
}

void drwnGaussianMixture::sample(vector<double> &x) const
{
    x.resize(_n);
    VectorXd xx(_n);
    this->sample(xx);
    Eigen::Map<VectorXd>(&x[0], x.size()) = xx;
}

// training
void drwnGaussianMixture::train(const vector<vector<double> >& x, double lambda)
{
    DRWN_ASSERT((x.size() > _g.size()) && (x[0].size() == _n));
    DRWN_FCN_TIC;

    // TODO: use method of Orchard and Bouman, 1991.

    // compute global covariance
    drwnSuffStats globalSuffStats(_n, DRWN_PSS_FULL);
    globalSuffStats.accumulate(x);

    // iterate expection-maximization
    if (drwnThreadPool::MAX_THREADS <= 1) {
        vector<double> y(_g.size());
        vector<drwnSuffStats> suffStats(_g.size(), drwnSuffStats(_n, DRWN_PSS_FULL));

        double lastLogLikelihood = -DRWN_DBL_MAX;
        for (int t = 0; t < MAX_ITERATIONS; t++) {
            // clear sufficient statistics
            for (unsigned k = 0; k < _g.size(); k++) {
                suffStats[k].clear();
            }

            // E-step
            double logLikelihood = 0.0;
            for (unsigned i = 0; i < x.size(); i++) {
                for (unsigned k = 0; k < _g.size(); k++) {
                    y[k] = _logLambda[k] + _g[k].evaluateSingle(x[i]);
                }
                const double logZ = drwn::expAndNormalize(y);
                
                // maintain sufficient statistics
                for (unsigned k = 0; k < _g.size(); k++) {
                    suffStats[k].accumulate(x[i], y[k]);
                }
                
                logLikelihood += logZ; //this->evaluateSingle(x[i]);
            }
            DRWN_LOG_STATUS("...GMM learning iteration " << t << "; log-likelihood " << logLikelihood << ";");
            if ((logLikelihood - lastLogLikelihood)
                <= DRWN_EPSILON * (fabs(logLikelihood) + fabs(lastLogLikelihood))) {
                DRWN_LOG_DEBUG("...converged");
                break;
            }
            lastLogLikelihood = logLikelihood;

            // M-step
            for (unsigned k = 0; k < _g.size(); k++) {
                // regularize sufficient statistics by global mean/sigma
                suffStats[k].accumulate(globalSuffStats, lambda);
                // update cluster mean and covariance
                _g[k].train(suffStats[k]);
                // update mixture weights
                _logLambda[k] = suffStats[k].count();
            }

            // normalize mixture weights
            _logLambda = (_logLambda / _logLambda.sum()).array().log();
        }

    } else {

        // create thread pool
        const unsigned nThreads = std::min((unsigned)x.size(), drwnThreadPool::MAX_THREADS);
        drwnThreadPool threadPool;
        vector<drwnGaussianMixtureThread *> jobs(nThreads, (drwnGaussianMixtureThread *)NULL);
        for (unsigned i = 0; i < nThreads; i++) {
            jobs[i] = new drwnGaussianMixtureThread(_n, x, &_g, &_logLambda, i, nThreads);
        }

        vector<drwnSuffStats> suffStats(_g.size(), drwnSuffStats(_n, DRWN_PSS_FULL));

        double lastLogLikelihood = -DRWN_DBL_MAX;
        for (int t = 0; t < MAX_ITERATIONS; t++) {

            // force caching of inverse covariance
            for (unsigned k = 0; k < _g.size(); k++) {
                _g[k].logPartitionFunction();
            }

            // E-step
            threadPool.start();
            for (unsigned i = 0; i < nThreads; i++) {
                threadPool.addJob(jobs[i]);
            }
            threadPool.finish();

            // clear sufficient statistics
            for (unsigned k = 0; k < _g.size(); k++) {
                suffStats[k].clear();
            }

            double logLikelihood = 0.0;
            for (unsigned i = 0; i < nThreads; i++) {
                logLikelihood += jobs[i]->logLikelihood;
                for (unsigned k = 0; k < _g.size(); k++) {
                    suffStats[k].accumulate(jobs[i]->stats[k]);
                }
            }

            DRWN_LOG_STATUS("...GMM learning iteration " << t << "; log-likelihood " << logLikelihood << ";");
            if ((logLikelihood - lastLogLikelihood)
                <= DRWN_EPSILON * (fabs(logLikelihood) + fabs(lastLogLikelihood))) {
                DRWN_LOG_DEBUG("...converged");
                break;
            }
            lastLogLikelihood = logLikelihood;

            // M-step
            for (unsigned k = 0; k < _g.size(); k++) {
                // regularize sufficient statistics by global mean/sigma
                suffStats[k].accumulate(globalSuffStats, lambda);
                // update cluster mean and covariance
                _g[k].train(suffStats[k]);
                // update mixture weights
                _logLambda[k] = suffStats[k].count();
            }

            // normalize mixture weights
            _logLambda = (_logLambda / _logLambda.sum()).array().log();
        }

        // free jobs
        for (unsigned i = 0; i < nThreads; i++) {
            delete jobs[i];
        }
    }

#if 0
    // DEBUGGING
    cout << _logLambda.transpose() << "\n\n";
    for (unsigned k = 0; k < _g.size(); k++) {
        _g[k].dump();
    }
#endif

    DRWN_FCN_TOC;
}

// standard operators
drwnGaussianMixture& drwnGaussianMixture::operator=(const drwnGaussianMixture& model)
{
    if (this != &model) {
        initialize(model.dimension(), model.mixtures());
        _logLambda = model._logLambda;
        for (unsigned i = 0; i < _g.size(); i++) {
            _g[i] = model._g[i];
        }
    }

    return *this;
}

// drwnGaussianMixtureConfig ------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnGaussianMixture
//! \b maxIterations :: maximum number of training iterations (default: 100)

class drwnGaussianMixtureConfig : public drwnConfigurableModule {
public:
    drwnGaussianMixtureConfig() : drwnConfigurableModule("drwnGaussianMixture") { }
    ~drwnGaussianMixtureConfig() { }

    void usage(ostream &os) const {
        os << "      maxIterations :: maximum number of training iterations (default: "
           << drwnGaussianMixture::MAX_ITERATIONS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxIterations")) {
            drwnGaussianMixture::MAX_ITERATIONS = std::max(0, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnGaussianMixtureConfig gGaussianMixtureConfig;

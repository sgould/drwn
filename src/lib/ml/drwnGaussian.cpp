/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGaussian.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <cmath>
#include <iterator>

// drwn libraries
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnSuffStats.h"
#include "drwnGaussian.h"

using namespace std;

// drwnGaussian -------------------------------------------------------------

bool drwnGaussian::AUTO_RIDGE = false;

drwnGaussian::drwnGaussian(int n) :
    _n(n), _invSigma(NULL), _logZ(0.0), _mL(NULL)
{
    initialize(n);
}

drwnGaussian::drwnGaussian(const VectorXd& mu, double sigma2) :
    _invSigma(NULL), _logZ(0.0), _mL(NULL)
{
    initialize(mu, sigma2);
}

drwnGaussian::drwnGaussian(const VectorXd& mu, const MatrixXd& sigma2) :
    _invSigma(NULL), _logZ(0.0), _mL(NULL)
{
    initialize(mu, sigma2);
}

drwnGaussian::drwnGaussian(const vector<double>& mu, double sigma2) :
    _invSigma(NULL), _logZ(0.0), _mL(NULL)
{
    initialize(Eigen::Map<const VectorXd>(&mu[0], mu.size()), sigma2);
}

drwnGaussian::drwnGaussian(const drwnSuffStats& stats) :
    _n(stats.size()), _invSigma(NULL), _logZ(0.0), _mL(NULL)
{
    initialize(_n);
    train(stats, DRWN_EPSILON);
}

drwnGaussian::drwnGaussian(const drwnGaussian& model) :
    _n(model._n), _mu(model._mu), _mSigma(model._mSigma), _invSigma(NULL),
    _logZ(model._logZ), _mL(NULL)
{
    if (model._invSigma != NULL) {
        _invSigma = new MatrixXd(*model._invSigma);
    }
    if (model._mL != NULL) {
        _mL = new MatrixXd(*model._mL);
    }
}

drwnGaussian::~drwnGaussian()
{
    freeCachedParameters();
}

// initialization
void drwnGaussian::initialize(int n)
{
    DRWN_ASSERT(n > 0);

    _n = n;
    freeCachedParameters();
    _mu = VectorXd::Zero(n);
    _mSigma = MatrixXd::Identity(n, n);
    _invSigma = new MatrixXd(MatrixXd::Identity(n, n));
    _logZ = -0.5f * log(2.0 * M_PI);
}

void drwnGaussian::initialize(const VectorXd& mu, double sigma2)
{
    DRWN_ASSERT(sigma2 > 0.0);

    _n = mu.rows();

    freeCachedParameters();
    _mu = mu;
    _mSigma = sigma2 * MatrixXd::Identity(_n, _n);

    _invSigma = new MatrixXd(MatrixXd::Identity(_n, _n) / sigma2);
    _logZ = -0.5 * (double)_n * log(2.0 * M_PI * sigma2);

    DRWN_ASSERT(!isnan(_logZ));
    DRWN_ASSERT(!isinf(_logZ));
}

void drwnGaussian::initialize(const VectorXd &mu, const MatrixXd &sigma2)
{
    _n = mu.rows();
    DRWN_ASSERT((sigma2.rows() == _n) && (sigma2.cols() == _n));

    freeCachedParameters();
    _mu = mu;
    _mSigma = sigma2;
    updateCachedParameters();
}

// marginalizing
drwnGaussian *drwnGaussian::marginalize(const vector<int>& indx) const
{
    DRWN_ASSERT(!indx.empty());

    drwnGaussian *g = new drwnGaussian(indx.size());

    for (unsigned i = 0; i < indx.size(); i++) {
        g->_mu(i) = _mu(indx[i]);
        for (unsigned j = 0; j < indx.size(); j++) {
            g->_mSigma(i, j) = _mSigma(indx[i], indx[j]);
	}
    }

    g->freeCachedParameters();
    return g;
}

// conditioning
drwnGaussian drwnGaussian::reduce(const vector<double>& x2,
    const vector<int>& indx2) const
{
    DRWN_ASSERT((x2.size() == indx2.size()) && (x2.size() < (unsigned)_n));

    int n1 = _n - (int)x2.size();
    int n2 = (int)x2.size();
    vector<int> indxAll(_n);
    for (int i = 0; i < _n; i++) {
        indxAll[i] = i;
    }
    vector<int> indx1;
    set_difference(indxAll.begin(), indxAll.end(),
        indx2.begin(), indx2.end(),
        insert_iterator<vector<int> >(indx1, indx1.end()));

    drwnGaussian g(n1);

    VectorXd mu2(n2);
    MatrixXd Sigma22(n2, n2);
    MatrixXd Sigma21(n2, n1);

    for (int i = 0; i < n2; i++) {
        DRWN_ASSERT((indx2[i] >= 0) && (indx2[i] < _n));
        mu2(i) = x2[i] - _mu(indx2[i]);
        for (int j = 0; j < n2; j++) {
            Sigma22(i, j) = _mSigma(indx2[i], indx2[j]);
        }
        for (int j = 0; j < n1; j++) {
            Sigma21(i, j) = _mSigma(indx2[i], indx1[j]);
        }
    }

    MatrixXd sigmaProduct(n2, n1);
    Eigen::LDLT<MatrixXd> cholesky(Sigma22);
    for (int i = 0; i < n1; i++) {
        sigmaProduct.col(i) = cholesky.solve(Sigma21.col(i));
    }

    g._mSigma = -1.0 * Sigma21.transpose() * sigmaProduct;
    g._mu = sigmaProduct.transpose() * mu2;

    for (int i = 0; i < n1; i++) {
        g._mu(i) += _mu(indx1[i]);
        for (int j = 0; j < n1; j++) {
            g._mSigma(i, j) += _mSigma(indx1[i], indx1[j]);
        }
    }

    g.updateCachedParameters();
    return g;
}

drwnGaussian drwnGaussian::reduce(const map<int, double>& x) const
{
    vector<int> indx2;
    vector<double> x2;

    indx2.reserve(x.size());
    x2.reserve(x.size());
    for (map<int, double>::const_iterator it = x.begin(); it != x.end(); ++it) {
        indx2.push_back(it->first);
        x2.push_back(it->second);
    }

    return this->reduce(x2, indx2);
}

drwnConditionalGaussian drwnGaussian::conditionOn(const vector<int>& indx2) const
{
    DRWN_ASSERT(indx2.size() < (unsigned)_n);

    int n1 = _n - (int)indx2.size();
    int n2 = (int)indx2.size();
    vector<int> indxAll(_n);
    for (int i = 0; i < _n; i++) {
        indxAll[i] = i;
    }
    vector<int> indx1;
    set_difference(indxAll.begin(), indxAll.end(),
        indx2.begin(), indx2.end(),
        insert_iterator<vector<int> >(indx1, indx1.end()));

    VectorXd mu2(n2);
    MatrixXd Sigma22(n2, n2);
    MatrixXd Sigma21(n2, n1);

    for (int i = 0; i < n2; i++) {
        DRWN_ASSERT((indx2[i] >= 0) && (indx2[i] < _n));
        mu2(i) = _mu(indx2[i]);
        for (int j = 0; j < n2; j++) {
            Sigma22(i, j) = _mSigma(indx2[i], indx2[j]);
        }
        for (int j = 0; j < n1; j++) {
            Sigma21(i, j) = _mSigma(indx2[i], indx1[j]);
        }
    }

    MatrixXd sigmaProduct(n2, n1);
    Eigen::LDLT<MatrixXd> cholesky(Sigma22);
    for (int i = 0; i < n1; i++) {
        sigmaProduct.col(i) = cholesky.solve(Sigma21.col(i));
    }

    VectorXd mu = -1.0 * sigmaProduct.transpose() * mu2;
    MatrixXd Sigma = -1.0 * Sigma21.transpose() * sigmaProduct;
    MatrixXd sigmaGain = sigmaProduct.transpose();

    return drwnConditionalGaussian(mu, Sigma, sigmaGain);
}

// i/o
bool drwnGaussian::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "n", toString(_n).c_str(), false);

    drwnXMLNode *node = drwnAddXMLChildNode(xml, "mu", NULL, false);
    drwnXMLUtils::serialize(*node, _mu);

    node = drwnAddXMLChildNode(xml, "sigma", NULL, false);
    drwnXMLUtils::serialize(*node, _mSigma);

    return true;
}

bool drwnGaussian::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "n") != NULL);
    initialize(atoi(drwnGetXMLAttribute(xml, "n")));

    drwnXMLNode *node = xml.first_node("mu");
    DRWN_ASSERT((node != NULL) && (node->next_sibling("mu") == NULL));
    drwnXMLUtils::deserialize(*node, _mu);
    DRWN_ASSERT(_mu.rows() == _n);

    node = xml.first_node("sigma");
    DRWN_ASSERT((node != NULL) && (node->next_sibling("sigma") == NULL));
    drwnXMLUtils::deserialize(*node, _mSigma);
    DRWN_ASSERT((_mSigma.rows() == _n) && (_mSigma.cols() == _n));

    updateCachedParameters();
    return true;
}

// evaluate (log-likelihood)
void drwnGaussian::evaluate(const MatrixXd& x, VectorXd& p) const
{
    DRWN_ASSERT((x.cols() == _n) && (x.rows() == p.rows()));

    guaranteeInvSigma();
    for (int i = 0; i < x.rows(); i++) {
        VectorXd z = x.row(i).transpose() - _mu;
        p(i) = -0.5 * (z.transpose() * (*_invSigma) * z)(0) + _logZ;
    }
}

void drwnGaussian::evaluate(const vector<vector<double> >& x, vector<double>& p) const
{
    DRWN_ASSERT(x.size() == p.size());

    guaranteeInvSigma();
    for (unsigned i = 0; i < x.size(); i++) {
        DRWN_ASSERT(x[i].size() == (unsigned)_n);
        VectorXd z = Eigen::Map<const VectorXd>(&x[i][0], _n) - _mu;
        p[i] = -0.5 * (z.transpose() * (*_invSigma) * z)(0) + _logZ;
    }
}

double drwnGaussian::evaluateSingle(const VectorXd& x) const
{
    DRWN_ASSERT(x.rows() == _n);
    guaranteeInvSigma();

    VectorXd z = x - _mu;
    return -0.5 * (z.transpose() * (*_invSigma) * z)(0) + _logZ;
}

double drwnGaussian::evaluateSingle(const vector<double>& x) const
{
    DRWN_ASSERT(x.size() == (unsigned)_n);
    guaranteeInvSigma();

    VectorXd z = Eigen::Map<const VectorXd>(&x[0], _n) - _mu;
    return -0.5 * (z.transpose() * (*_invSigma) * z)(0) + _logZ;
}

double drwnGaussian::evaluateSingle(double x) const
{
    DRWN_ASSERT(_n == 1);
    guaranteeInvSigma();

    double z = (x - _mu(0));
    return -0.5 * (*_invSigma)(0) * z * z + _logZ;
}

// sampling
void drwnGaussian::sample(VectorXd &x) const
{
    DRWN_ASSERT(x.rows() == _n);

    if (_mL == NULL) {
        _mL = new MatrixXd(_mSigma.llt().matrixL());
    }

    // box-muller sampling algorithm
    Vector2d uv;
    for (int i = 0; i < x.size() - 1; i += 2) {
        uv.setRandom();
        double s = uv.dot(uv);
        while (s >= 1.0 || s <= 0.0) {
            uv.setRandom();
            s = uv.dot(uv);
	}
        double scale = sqrt(-2.0 * log(s) / s);
        x[i] = scale * uv[0];
        x[i + 1] = scale * uv[1];
    }

    if (x.size() % 2 == 1) {
        uv.setRandom();
        double s = uv.dot(uv);
        while (s >= 1.0 || s <= 0.0) {
            uv.setRandom();
            s = uv.dot(uv);
	}
        double scale = sqrt(-2.0 * log(s) / s);
        x[x.size() - 1] = scale * uv[0];
    }

    // rescale unit gaussian
    x = (*_mL) * x + _mu;
}

void drwnGaussian::sample(vector<double>& x) const
{
    x.resize(_n);
    VectorXd xx(_n);
    sample(xx);
    Eigen::Map<VectorXd>(&x[0], x.size()) = xx;
}

// learn parameters
void drwnGaussian::train(const MatrixXd& x, double lambda)
{
    DRWN_ASSERT((x.cols() == _n) && ((x.rows() > 1) || (lambda > 0.0)));

    // accumulate statistics
    _mu = x.colwise().sum().transpose() / (double)x.rows();
    _mSigma = (x.transpose() * x) / (double)x.rows() -
        _mu.transpose() * _mu + lambda * MatrixXd::Identity(_n, _n);

    updateCachedParameters();
}

void drwnGaussian::train(const vector<vector<double> >& x, double lambda)
{
    DRWN_ASSERT((x.size() > 1) || (lambda > 0.0));

    _mu.setZero();
    _mSigma.setZero();

    // accumulate statistics
    for (int i = 0; i < (int)x.size(); i++) {
        DRWN_ASSERT(x[i].size() == (unsigned)_n);
        for (int j = 0; j < _n; j++) {
            _mu(j) += x[i][j];
            for (int k = 0; k <= j; k++) {
                _mSigma(j, k) += x[i][j] * x[i][k];
            }
        }
    }

    _mu /= (double)x.size();

    // normalize
    for (int j = 0; j < _n; j++) {
        for (int k = 0; k <= j; k++) {
            _mSigma(j, k) = _mSigma(j, k) / (double)x.size() - _mu(j) * _mu(k);
        }
        _mSigma(j, j) += lambda;
    }

    // copy symmetric part
    for (int j = 0; j < _n; j++) {
        for (int k = j + 1; k < _n; k++) {
            _mSigma(j, k) = _mSigma(k, j);
        }
    }

    updateCachedParameters();
}

// univariate training
void drwnGaussian::train(const vector<double> &x, double lambda)
{
    DRWN_ASSERT((_n == 1) && (x.size() > 1));

    _mu.setZero();
    _mSigma.setZero();

    // accumulate statistics
    for (int i = 0; i < (int)x.size(); i++) {
        _mu(0) += x[i];
        _mSigma(0, 0) += x[i] * x[i];
    }

    // normalize
    _mu /= (double)x.size();
    _mSigma = (_mSigma / (double)x.size() - _mu * _mu.transpose()).array() + lambda;

    updateCachedParameters();
}

void drwnGaussian::train(const drwnSuffStats& stats, double lambda)
{
    DRWN_ASSERT_MSG(stats.size() == _n, stats.size() << " != " << _n);

    // normalize sufficient statistics
    _mu = stats.firstMoments() / stats.count();
    if (stats.isDiagonal()) {
        _mSigma = stats.secondMoments().array() / stats.count();
        _mSigma -= _mu.array().square().matrix().asDiagonal();
        _mSigma += lambda * MatrixXd::Identity(_n, _n);
    } else {
        _mSigma = stats.secondMoments() / stats.count() -
            _mu * _mu.transpose() + lambda * MatrixXd::Identity(_n, _n);
    }

    updateCachedParameters();
}

double drwnGaussian::logPartitionFunction() const
{
    guaranteeInvSigma();
    return _logZ;
}

// kl-divergence, KL(*this || model)
// TODO: test
double drwnGaussian::klDivergence(const drwnGaussian& model) const
{
    DRWN_ASSERT(model._n == _n);
    model.guaranteeInvSigma();

    VectorXd z = model._mu - _mu;
    double s = (z.transpose() * (*_invSigma) * z)(0);
    double d = _logZ - model._logZ +
        0.5 * ((model._invSigma->cwiseProduct(_mSigma)).sum() + s - (double)_n);

    return d;
}

double drwnGaussian::klDivergence(const drwnSuffStats& stats) const
{
    DRWN_ASSERT(stats.size() == _n);
    return klDivergence(drwnGaussian(stats));
}

// standard operators
drwnGaussian& drwnGaussian::operator=(const drwnGaussian& model)
{
    if (this != &model) {
        freeCachedParameters();
        _n = model._n;
        _mu = model._mu;
        _mSigma = model._mSigma;
        if (model._invSigma != NULL) {
            _invSigma = new MatrixXd(*model._invSigma);
        }
        if (model._mL != NULL) {
            _mL = new MatrixXd(*model._mL);
        }
        _logZ = model._logZ;
    }

    return *this;
}

// protected functions
void drwnGaussian::freeCachedParameters()
{
    if (_invSigma != NULL) delete _invSigma;
    _invSigma = NULL;
    if (_mL != NULL) delete _mL;
    _mL = NULL;
}

void drwnGaussian::updateCachedParameters()
{
    freeCachedParameters();
}

inline void drwnGaussian::guaranteeInvSigma() const
{
    if (_invSigma == NULL) {
        _invSigma = new MatrixXd(_n, _n);

        double det = _mSigma.determinant();
        if ((!isfinite(det) || (det <= 0.0)) && (drwnLogger::getLogLevel() >= DRWN_LL_DEBUG)) {
            DRWN_LOG_DEBUG("drwnGaussian::_mSigma =");
            cout << _mSigma << "\n";
        }

        if ((det <= _n * DRWN_DBL_MIN) && (drwnGaussian::AUTO_RIDGE)) {
            MatrixXd Sigma(_mSigma);
            while (det <= _n * DRWN_DBL_MIN) {
                DRWN_LOG_ERROR("using auto-ridge regression for |Sigma| = " << det);
                double delta = exp(log(_n * DRWN_EPSILON - det) / (double)_n + DRWN_DBL_MIN) + DRWN_EPSILON;
                for (int i = 0; i < _n; i++) {
                    Sigma(i, i) += delta;
                    if (Sigma(i, i) < DRWN_EPSILON) {
                        Sigma(i, i) = DRWN_EPSILON;
                    }
                }
                det = Sigma.determinant();
            }
            *_invSigma = Sigma.inverse();
        } else {
            *_invSigma = _mSigma.inverse();
        }

        DRWN_ASSERT_MSG(isfinite(det) && (det > 0.0), "|Sigma| = " << det);
        _logZ = -0.5 * (float)_n * log(2.0 * M_PI) - 0.5 * log(det);
    }
}

// drwnConditionalGaussian --------------------------------------------------------

drwnConditionalGaussian::drwnConditionalGaussian(const VectorXd& mu, const MatrixXd &Sigma,
    const MatrixXd& SigmaGain) :
    _mu(mu), _mSigma(Sigma), _mSigmaGain(SigmaGain)
{
    _n = SigmaGain.rows();
    _m = SigmaGain.cols();
    DRWN_ASSERT(mu.rows() == _n);
    DRWN_ASSERT((Sigma.rows() == _n) && (Sigma.cols() == _n));
}

drwnConditionalGaussian::drwnConditionalGaussian(const drwnConditionalGaussian& model) :
    _n(model._n), _m(model._m), _mu(model._mu), _mSigma(model._mSigma), _mSigmaGain(model._mSigmaGain)
{
    // do nothing
}

drwnConditionalGaussian::~drwnConditionalGaussian()
{
    // do nothing
}

// i/o
bool drwnConditionalGaussian::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "n", toString(_n).c_str(), false);
    drwnAddXMLAttribute(xml, "m", toString(_m).c_str(), false);

    drwnXMLNode *node = drwnAddXMLChildNode(xml, "mu", NULL, false);
    drwnXMLUtils::serialize(*node, _mu);

    node = drwnAddXMLChildNode(xml, "sigma", NULL, false);
    drwnXMLUtils::serialize(*node, _mSigma);

    node = drwnAddXMLChildNode(xml, "gain", NULL, false);
    drwnXMLUtils::serialize(*node, _mSigmaGain);

    return true;
}

bool drwnConditionalGaussian::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "n") != NULL);
    _n = atoi(drwnGetXMLAttribute(xml, "n"));

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "m") != NULL);
    _m = atoi(drwnGetXMLAttribute(xml, "m"));

    drwnXMLNode *node = xml.first_node("mu");
    DRWN_ASSERT((node != NULL) && (node->next_sibling("mu") == NULL));
    drwnXMLUtils::deserialize(*node, _mu);
    DRWN_ASSERT(_mu.rows() == _n);

    node = xml.first_node("sigma");
    DRWN_ASSERT((node != NULL) && (node->next_sibling("sigma") == NULL));
    drwnXMLUtils::deserialize(*node, _mSigma);
    DRWN_ASSERT((_mSigma.rows() == _n) && (_mSigma.cols() == _n));

    node = xml.first_node("gain");
    DRWN_ASSERT((node != NULL) && (node->next_sibling("gain") == NULL));
    drwnXMLUtils::deserialize(*node, _mSigmaGain);
    DRWN_ASSERT((_mSigmaGain.rows() == _n) && (_mSigmaGain.cols() == _m));

    return true;
}

drwnGaussian drwnConditionalGaussian::reduce(const VectorXd& x)
{
    DRWN_ASSERT(x.rows() == _m);
    return drwnGaussian(_mSigmaGain * x + _mu, _mSigma);
}

drwnGaussian drwnConditionalGaussian::reduce(const vector<double>& x)
{
    return reduce(Eigen::Map<const VectorXd>(&x[0], x.size()));
}

// drwnGaussianConfig -------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnGaussian
//! \b autoRidge :: apply ridge-regression automatically if log-det too small

class drwnGaussianConfig : public drwnConfigurableModule {
public:
    drwnGaussianConfig() : drwnConfigurableModule("drwnGaussian") { }
    ~drwnGaussianConfig() { }

    void usage(ostream &os) const {
        os << "      autoRidge       :: apply ridge-regression automatically if log-det too small\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "autoRidge")) {
            drwnGaussian::AUTO_RIDGE = drwn::trueString(string(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnGaussianConfig gGaussianConfig;

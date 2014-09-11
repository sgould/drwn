/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnBoostedClassifier.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnBoostedClassifier.h"

using namespace std;
using namespace Eigen;

// drwnBoostedClassifier static members --------------------------------------

drwnBoostingMethod drwnBoostedClassifier::METHOD = DRWN_BOOST_DISCRETE;
int drwnBoostedClassifier::NUM_ROUNDS = 100;
int drwnBoostedClassifier::MAX_DEPTH = 2;
double drwnBoostedClassifier::SHRINKAGE = 0.95;

// drwnBoostedClassifier -----------------------------------------------------

drwnBoostedClassifier::drwnBoostedClassifier() :
    drwnClassifier(), _method(METHOD), _numRounds(NUM_ROUNDS),
    _maxDepth(MAX_DEPTH), _shrinkage(SHRINKAGE)
{
    // define properties
    declareProperty("numRounds", new drwnRangeProperty(&_numRounds, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("shrinkage", new drwnDoubleRangeProperty(&_shrinkage));
}

drwnBoostedClassifier::drwnBoostedClassifier(unsigned n, unsigned k) :
    drwnClassifier(n, k), _method(METHOD), _numRounds(NUM_ROUNDS),
    _maxDepth(MAX_DEPTH), _shrinkage(SHRINKAGE)
{
    // define properties
    declareProperty("numRounds", new drwnRangeProperty(&_numRounds, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("shrinkage", new drwnDoubleRangeProperty(&_shrinkage));

    initialize(n, k);
}

drwnBoostedClassifier::drwnBoostedClassifier(const drwnBoostedClassifier &c) :
    drwnClassifier(c),  _method(c._method), _numRounds(c._numRounds),
    _maxDepth(c._maxDepth), _shrinkage(c._shrinkage), _alphas(c._alphas)
{
    // define properties
    declareProperty("numRounds", new drwnRangeProperty(&_numRounds, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("shrinkage", new drwnDoubleRangeProperty(&_shrinkage));

    // deep copy weak learners
    _weakLearners.resize(c._weakLearners.size(), NULL);
    for (unsigned i = 0; i < c._weakLearners.size(); i++) {
        _weakLearners[i] = new drwnDecisionTree(*c._weakLearners[i]);
    }
}

drwnBoostedClassifier::~drwnBoostedClassifier()
{
    for (unsigned i = 0; i < _weakLearners.size(); i++) {
        delete _weakLearners[i];
    }
}

// initialization
void drwnBoostedClassifier::initialize(unsigned n, unsigned k)
{
    drwnClassifier::initialize(n, k);
    for (unsigned i = 0; i < _weakLearners.size(); i++) {
        delete _weakLearners[i];
    }
    _weakLearners.clear();
    _alphas.clear();
}

// i/o
bool drwnBoostedClassifier::save(drwnXMLNode& xml) const
{
    drwnClassifier::save(xml);

    for (unsigned i = 0; i < _weakLearners.size(); i++) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "weakLearner", NULL, false);
        drwnAddXMLAttribute(*node, "weight", toString(_alphas[i]).c_str(), false);
        _weakLearners[i]->save(*node);
    }

    return true;
}

bool drwnBoostedClassifier::load(drwnXMLNode& xml)
{
    drwnClassifier::load(xml);

    for (drwnXMLNode *node = xml.first_node("weakLearner"); node != NULL;
         node = node->next_sibling("weakLearner")) {
        _alphas.push_back(atof(drwnGetXMLAttribute(*node, "weight")));
        _weakLearners.push_back(new drwnDecisionTree());
        _weakLearners.back()->load(*node);
    }

    return true;
}

// training
double drwnBoostedClassifier::train(const drwnClassifierDataset& dataset)
{
    DRWN_FCN_TIC;

    // pre-compute sorted feature index
    vector<vector<int> > sortIndex;
    if (drwnDecisionTree::CACHE_SORTED_INDEXES) {
        drwnDecisionTree::computeSortedFeatureIndex(dataset.features, sortIndex);
    }

    // allocate weights
    vector<double> weights;
    if (dataset.hasWeights()) {
        weights = dataset.weights;
    } else {
        weights.resize(dataset.size(), 1.0);
    }

    // mark unknown samples
    drwnBitArray sampleIndex(dataset.size());
    sampleIndex.ones();
    for (int i = 0; i < dataset.size(); i++) {
        if (dataset.targets[i] < 0) {
            sampleIndex.clear(i);
            weights[i] = 0.0;
        }
    }

    vector<int> predicted(dataset.size(), -1);

    // iterate over rounds
    DRWN_START_PROGRESS("training", _numRounds);
    for (int i = 0; i < _numRounds; i++) {
        DRWN_LOG_STATUS("training boosted classifier round " << i << " of " << _numRounds << "...");
        DRWN_INC_PROGRESS;

        // normalize the sample weights
        Eigen::Map<VectorXd>(&weights[0], weights.size()) /=
            Eigen::Map<VectorXd>(&weights[0], weights.size()).sum();

        // learn a weak-learner with current weights
        drwnDecisionTree *tree = new drwnDecisionTree(_nFeatures, _nClasses);
        tree->setProperty(tree->findProperty("maxDepth"), _maxDepth);
        tree->learnDecisionTree(dataset.features, dataset.targets,
            weights, sortIndex, sampleIndex);

        // predict classes and calculate eplison
        double epsilon = 0.0;
        for (int j = 0; j < dataset.size(); j++) {
            if (dataset.targets[j] < 0) continue;
            predicted[j] = tree->getClassification(dataset.features[j]);
            if (predicted[j] != dataset.targets[j]) {
                epsilon += weights[j];
            }
        }

        if (epsilon >= 1.0 - 1.0 / _nClasses) {
            DRWN_LOG_WARNING("boosting terminated at round "
                << (i + 1) << " of " << _numRounds);
            delete tree;
            break;
        }

        // check for perfect classification
        if ((i == 0) && (epsilon == 0.0)) {
            DRWN_LOG_WARNING("boosting found a perfect classifier in first round");
            _alphas.push_back(1.0);
            _weakLearners.push_back(tree);
            break;
        }

        // calculate boosting coefficient
        double alpha;
        if (_method == DRWN_BOOST_GENTLE) {
            alpha = 1.0;
        } else {
            alpha = log((1.0 - epsilon) / epsilon) + log(_nClasses - 1.0);
            if (!isfinite(alpha)) {
                DRWN_LOG_WARNING("boosting terminated at round "
                    << (i + 1) << " of " << _numRounds << " (non-finite alpha)");
                delete tree;
                break;
            }
        }

        _alphas.push_back(alpha);
        _weakLearners.push_back(tree);

        // update the sample weights
        const double nu = exp(alpha);
        for (unsigned j = 0; j < weights.size(); j++) {
            if (predicted[j] != dataset.targets[j]) {
                weights[j] *= nu;
            }
        }
    }
    DRWN_END_PROGRESS;

#if 0
    // normalize boosting coefficients
    Eigen::Map<VectorXd>(&_alphas[0], _alphas.size()) /=
        Eigen::Map<VectorXd>(&_alphas[0], _alphas.size()).sum();
#endif

    _bValid = true;

    // return classification accuracy
    double totalCorrect = 0.0;
    double totalWeight = 0.0;
    for (int j = 0; j < dataset.size(); j++) {
        if (dataset.targets[j] < 0) continue;
        predicted[j] = this->getClassification(dataset.features[j]);
        if (predicted[j] == dataset.targets[j]) {
            totalCorrect += dataset.hasWeights() ? dataset.weights[j] : 1.0;
        }
        totalWeight += dataset.hasWeights() ? dataset.weights[j] : 1.0;
    }

    DRWN_FCN_TOC;
    return (totalWeight > 0.0) ? totalCorrect / totalWeight : 1.0;
}

void drwnBoostedClassifier::pruneRounds(unsigned numRounds)
{
    DRWN_ASSERT(numRounds > 0);
    if (numRounds < _alphas.size()) {
        for (unsigned i = numRounds; i < _weakLearners.size(); i++) {
            delete _weakLearners[i];
        }
        _weakLearners.resize(numRounds);
        _alphas.resize(numRounds);
    }
}

// evaluation (log-probability)
void drwnBoostedClassifier::getClassScores(const vector<double>& features,
    vector<double>& outputScores) const
{
    DRWN_ASSERT_MSG((int)features.size() == _nFeatures, (int)features.size() << " != " << _nFeatures);

    // initialize output scores
    outputScores.resize(_nClasses);
    fill(outputScores.begin(), outputScores.end(), 0.0);

    // iterate over weak learners
    /* vector<double> weakLearnerScores; */
    for (unsigned i = 0; i < _weakLearners.size(); i++) {
#if 0
        /*
        _weakLearners[i]->getClassScores(features, weakLearnerScores);
        Eigen::Map<VectorXd>(&outputScores[0], outputScores.size()) +=
            _alphas[i] * Eigen::Map<VectorXd>(&weakLearnerScores[0], weakLearnerScores.size());
        */
        int predictedClass = _weakLearners[i]->getClassification(features);
        outputScores[predictedClass] += _alphas[i];
#else
        // fast version (avoids function calls)
        const drwnDecisionTree *p = _weakLearners[i];
        while (p->_splitIndx >= 0) {
            p = (features[p->_splitIndx] < p->_splitValue) ? p->_leftChild : p->_rightChild;
        }

        outputScores[p->_predictedClass] += _alphas[i];
#endif
    }
}

// drwnBoostedClassifierConfig ----------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnBoostedClassifier
//! \b method :: boosting method (DISCRETE (default), GENTLE, or REAL)\n
//! \b numRounds :: maximum number of boosting rounds (default: 100)\n
//! \b maxDepth :: maximum depth of each decision tree (default: 2)\n
//! \b skrinkage :: boosting shrinkage (default: 0.95)\n

class drwnBoostedClassifierConfig : public drwnConfigurableModule {
public:
    drwnBoostedClassifierConfig() : drwnConfigurableModule("drwnBoostedClassifier") { }
    ~drwnBoostedClassifierConfig() { }

    void usage(ostream &os) const {
        os << "      method        :: boosting method (DISCRETE (default), GENTLE, or REAL)\n";
        os << "      numRounds     :: maximum number of boosting rounds (default: "
           << drwnBoostedClassifier::NUM_ROUNDS << ")\n";
        os << "      maxDepth      :: maximum depth of each decision tree (default: "
           << drwnBoostedClassifier::MAX_DEPTH << ")\n";
        os << "      skrinkage     :: boosting shrinkage (default: "
           << drwnBoostedClassifier::SHRINKAGE << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "method")) {
            if (!strcasecmp(value, "DISCRETE")) {
                drwnBoostedClassifier::METHOD = DRWN_BOOST_DISCRETE;
            } else if (!strcasecmp(value, "GENTLE")) {
                drwnBoostedClassifier::METHOD = DRWN_BOOST_GENTLE;
            } else if (!strcasecmp(value, "REAL")) {
                drwnBoostedClassifier::METHOD = DRWN_BOOST_REAL;
            } else {
                DRWN_LOG_FATAL("unrecognized configuration value " << value
                    << " for option " << name << " in " << this->name());
            }
        } else if (!strcmp(name, "numRounds")) {
            drwnBoostedClassifier::NUM_ROUNDS = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxDepth")) {
            drwnBoostedClassifier::MAX_DEPTH = std::max(1, atoi(value));
        } else if (!strcmp(name, "shrinkage")) {
            drwnBoostedClassifier::SHRINKAGE = std::max(0.0, std::min(1.0, atof(value)));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnBoostedClassifierConfig gBoostedClassifierConfig;

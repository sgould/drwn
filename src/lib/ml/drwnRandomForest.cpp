/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRandomForest.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnRandomForest.h"

using namespace std;
using namespace Eigen;

// drwnRandomForest static members -------------------------------------------

int drwnRandomForest::NUM_TREES = 100;
int drwnRandomForest::MAX_DEPTH = 2;
int drwnRandomForest::MAX_FEATURES = 10;

// drwnRandomForest ----------------------------------------------------------

drwnRandomForest::drwnRandomForest() : drwnClassifier(),
    _numTrees(NUM_TREES), _maxDepth(MAX_DEPTH), _maxFeatures(MAX_FEATURES)
{
    // define properties
    declareProperty("numTrees", new drwnRangeProperty(&_numTrees, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("maxFeatures", new drwnRangeProperty(&_maxFeatures, 1, DRWN_INT_MAX));
}

drwnRandomForest::drwnRandomForest(unsigned n, unsigned k) : drwnClassifier(n, k),
    _numTrees(NUM_TREES), _maxDepth(MAX_DEPTH), _maxFeatures(MAX_FEATURES)
{
    // define properties
    declareProperty("numTrees", new drwnRangeProperty(&_numTrees, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("maxFeatures", new drwnRangeProperty(&_maxFeatures, 1, DRWN_INT_MAX));

    initialize(n, k);
}

drwnRandomForest::drwnRandomForest(const drwnRandomForest &c) : drwnClassifier(c),
    _numTrees(c._numTrees), _maxDepth(c._maxDepth), _maxFeatures(c._maxFeatures),
    _alphas(c._alphas)
{
    // define properties
    declareProperty("numTrees", new drwnRangeProperty(&_numTrees, 1, DRWN_INT_MAX));
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
    declareProperty("maxFeatures", new drwnRangeProperty(&_maxFeatures, 1, DRWN_INT_MAX));

    // deep copy decision trees
    _forest.resize(c._forest.size(), NULL);
    for (unsigned i = 0; i < c._forest.size(); i++) {
        _forest[i] = new drwnDecisionTree(*c._forest[i]);
    }
}

drwnRandomForest::~drwnRandomForest()
{
    for (unsigned i = 0; i < _forest.size(); i++) {
        delete _forest[i];
    }
}

// initialization
void drwnRandomForest::initialize(unsigned n, unsigned k)
{
    drwnClassifier::initialize(n, k);
    for (unsigned i = 0; i < _forest.size(); i++) {
        delete _forest[i];
    }
    _forest.clear();
    _alphas.clear();
}

// i/o
bool drwnRandomForest::save(drwnXMLNode& xml) const
{
    drwnClassifier::save(xml);

    for (unsigned i = 0; i < _forest.size(); i++) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnDecisionTree", NULL, false);
        drwnAddXMLAttribute(*node, "weight", toString(_alphas[i]).c_str(), false);
        _forest[i]->save(*node);
    }

    return true;
}

bool drwnRandomForest::load(drwnXMLNode& xml)
{
    drwnClassifier::load(xml);

    for (drwnXMLNode *node = xml.first_node("drwnDecisionTree"); node != NULL; node = node->next_sibling("drwnDecisionTree")) {
        _alphas.push_back(atof(drwnGetXMLAttribute(*node, "weight")));
        _forest.push_back(new drwnDecisionTree());
        _forest.back()->load(*node);
    }

    return true;
}

// training
double drwnRandomForest::train(const drwnClassifierDataset& dataset)
{
    DRWN_FCN_TIC;

    // pre-compute sorted feature index
    vector<vector<int> > sortIndex;
    drwnDecisionTree::computeSortedFeatureIndex(dataset.features, sortIndex);

    // iterate over rounds
    drwnBitArray sampleIndex(dataset.size());
    DRWN_START_PROGRESS("training", _numTrees);
    for (int i = 0; i < _numTrees; i++) {
        DRWN_LOG_STATUS("training decision tree classifier " << i << " of " << _numTrees << "...");
        DRWN_INC_PROGRESS;

        // randomly choose training examples
        vector<double> weights(dataset.size(), 0.0);
        sampleIndex.zeros();
        for (int j = 0; j < dataset.size(); j++) {
            int indx = rand() % dataset.size();
            while (dataset.targets[indx] < 0) {
                // TODO: detect infinite loop
                indx = rand() % dataset.size();
            }

            sampleIndex.set(indx);
            weights[indx] += (dataset.hasWeights() ? dataset.weights[indx] : 1.0);
        }

        // TODO: randomly choose features


        // learn classifier
        drwnDecisionTree *tree = new drwnDecisionTree(_nFeatures, _nClasses);
        tree->setProperty(tree->findProperty("maxDepth"), _maxDepth);
        tree->learnDecisionTree(dataset.features, dataset.targets,
            weights, sortIndex, sampleIndex);

        // TODO: evaluate quality

        _alphas.push_back(1.0);
        _forest.push_back(tree);
    }
    DRWN_END_PROGRESS;

    _bValid = true;

    // return classification accuracy
    double totalCorrect = 0.0;
    double totalWeight = 0.0;
    for (int j = 0; j < dataset.size(); j++) {
        if (dataset.targets[j] < 0) continue;
        int predicted = this->getClassification(dataset.features[j]);
        if (predicted == dataset.targets[j]) {
            totalCorrect += dataset.hasWeights() ? dataset.weights[j] : 1.0;
        }
        totalWeight += dataset.hasWeights() ? dataset.weights[j] : 1.0;
    }

    DRWN_FCN_TOC;
    return (totalWeight > 0.0) ? totalCorrect / totalWeight : 1.0;
}

// evaluation (log-probability)
void drwnRandomForest::getClassScores(const vector<double>& features,
    vector<double>& outputScores) const
{
    DRWN_ASSERT((int)features.size() == _nFeatures);

    // initialize output scores
    outputScores.resize(_nClasses, 0.0);
    fill(outputScores.begin(), outputScores.end(), 0.0);

    // iterate over ensemble
    for (unsigned i = 0; i < _forest.size(); i++) {
        // fast version (avoids function calls)
        const drwnDecisionTree *p = _forest[i];
        while (p->_splitIndx >= 0) {
            p = (features[p->_splitIndx] < p->_splitValue) ? p->_leftChild : p->_rightChild;
        }
        outputScores[p->_predictedClass] += _alphas[i];
    }
}

// drwnRandomForestConfig ----------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnRandomForest
//! \b numTrees :: maximum number of trees in the forest (default: 100)\n
//! \b maxDepth :: maximum depth of each decision tree (default: 2)\n
//! \b maxFeatures :: number of features for each tree (default: 10)

class drwnRandomForestConfig : public drwnConfigurableModule {
public:
    drwnRandomForestConfig() : drwnConfigurableModule("drwnRandomForest") { }
    ~drwnRandomForestConfig() { }

    void usage(ostream &os) const {
        os << "      numTrees      :: maximum number of trees in the forest (default: "
           << drwnRandomForest::NUM_TREES << ")\n";
        os << "      maxDepth      :: maximum depth of each decision tree (default: "
           << drwnRandomForest::MAX_DEPTH << ")\n";
        os << "      maxFeatures   :: number of features for each tree (default: "
           << drwnRandomForest::MAX_FEATURES << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "numTrees")) {
            drwnRandomForest::NUM_TREES = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxDepth")) {
            drwnRandomForest::MAX_DEPTH = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxFeatures")) {
            drwnRandomForest::MAX_FEATURES = std::max(1, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnRandomForestConfig gRandomForestConfig;

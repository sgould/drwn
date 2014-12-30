/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDecisionTree.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnDecisionTree.h"

using namespace std;
using namespace Eigen;

// drwnDecisionTreeThread ---------------------------------------------------

class drwnDecisionTreeThread : public drwnThreadJob {
protected:
    const set<int> _featureSet;
    const vector<vector<double> >& _x;
    const vector<int>& _y;
    const vector<double>& _w;
    const vector<vector<int> >& _sortIndex;
    const drwnBitArray& _sampleIndex;
    const vector<double>& _classCounts;
    const double _H;
    const double _totalWeight;

public:
    int bestFeature;
    double bestScore;
    double bestSplit;

public:
    drwnDecisionTreeThread(const set<int>& featureSet, const vector<vector<double> > &x,
        const vector<int>& y, const vector<double>& w, const vector<vector<int> >& sortIndex,
        const drwnBitArray& sampleIndex, const vector<double>& classCounts,
        double H, double totalWeight) :
        _featureSet(featureSet), _x(x), _y(y), _w(w), _sortIndex(sortIndex),
        _sampleIndex(sampleIndex), _classCounts(classCounts),
        _H(H), _totalWeight(totalWeight),
        bestFeature(-1), bestScore(0.0), bestSplit(0.0) { /* do nothing */ }
    ~drwnDecisionTreeThread() { /* do nothing */ }

    void operator()() {
        const int numClasses = (int)_classCounts.size();

        for (set<int>::const_iterator it = _featureSet.begin(); it != _featureSet.end(); ++it) {
            vector<int> indx;

            // check if sorted indexes have been pre-computed
            if (_sortIndex.empty()) {
                drwnDecisionTree::computeSortedFeatureIndex(_x, _sampleIndex, *it, indx);
            } else {
                indx.reserve(_sortIndex[*it].size());
                for (vector<int>::const_iterator jt = _sortIndex[*it].begin(); jt != _sortIndex[*it].end(); ++jt) {
                    if (_sampleIndex[*jt]) {
                        indx.push_back(*jt);
                    }
                }
            }

            const int numSamples = (int)indx.size();
            if (numSamples < drwnDecisionTree::MIN_SAMPLES) continue;

            vector<double> classCountsLeft(numClasses, 0.0);
            vector<double> classCountsRight(_classCounts);

            int t = 0;
            while (t < numSamples - 1) {

                // find next threshold
                const int nextIndex = std::min(numSamples - 1,
                    t + numSamples / drwnDecisionTree::MAX_FEATURE_THRESHOLDS + 1);
                while (t != nextIndex) {
                    classCountsLeft[_y[indx[t]]] += _w[indx[t]];
                    t += 1;
                }

                while ((t != numSamples - 1) && (_x[indx[t]][*it] == _x[indx[t - 1]][*it])) {
                    classCountsLeft[_y[indx[t]]] += _w[indx[t]];
                    t += 1;
                }

                // TODO: is this needed?
                if (t == numSamples - 1) break;

                const double leftWeight = Eigen::Map<VectorXd>(&classCountsLeft[0], numClasses).sum();
                Eigen::Map<ArrayXd>(&classCountsRight[0], numClasses) =
                    Eigen::Map<const ArrayXd>(&_classCounts[0], numClasses) -
                    Eigen::Map<const ArrayXd>(&classCountsLeft[0], numClasses);

                // score threshold (information gain)
                double score;
                switch (drwnDecisionTree::SPLIT_CRITERION) {
                case DRWN_DT_SPLIT_ENTROPY:
                    score = _H - leftWeight * drwn::entropy(classCountsLeft) -
                        (_totalWeight - leftWeight) * drwn::entropy(classCountsRight);
                    break;
                case DRWN_DT_SPLIT_MISCLASS:
                    score = _H - (_totalWeight -
                        drwn::maxElem(classCountsLeft) - drwn::maxElem(classCountsRight));
                    break;
                case DRWN_DT_SPLIT_GINI:
                    score = _H - leftWeight * drwn::gini(classCountsLeft) -
                        (_totalWeight - leftWeight) * drwn::gini(classCountsRight);
                    break;
                default:
                    score = 0.0;
                    DRWN_LOG_FATAL("unknown split criterion " << drwnDecisionTree::SPLIT_CRITERION);
                }

                if (score > bestScore) {
                    bestFeature = *it;
                    bestScore = score;
                    bestSplit = 0.5 * (_x[indx[t]][*it] + _x[indx[t - 1]][*it]);
                }
            }
        }
    }
};

// drwnDecisionTree statics -------------------------------------------------

int drwnDecisionTree::MAX_DEPTH = 1;
int drwnDecisionTree::MAX_FEATURE_THRESHOLDS = 1000;
int drwnDecisionTree::MIN_SAMPLES = 0;
double drwnDecisionTree::LEAKAGE = 0.0;
drwnTreeSplitCriterion drwnDecisionTree::SPLIT_CRITERION = DRWN_DT_SPLIT_ENTROPY;
bool drwnDecisionTree::CACHE_SORTED_INDEXES = true;

// drwnDecisionTree ---------------------------------------------------------

drwnDecisionTree::drwnDecisionTree() :
    drwnClassifier(), _splitIndx(-1), _splitValue(0.0),
    _leftChild(NULL), _rightChild(NULL), _predictedClass(-1), _maxDepth(MAX_DEPTH)
{
    // define properties
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));
}

drwnDecisionTree::drwnDecisionTree(unsigned n, unsigned k) :
    drwnClassifier(n, k), _splitIndx(-1), _splitValue(0.0),
    _leftChild(NULL), _rightChild(NULL), _predictedClass(-1), _maxDepth(MAX_DEPTH)
{
    // define properties
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));

    initialize(n, k);
}

drwnDecisionTree::drwnDecisionTree(const drwnDecisionTree &c) :
    drwnClassifier(c), _splitIndx(c._splitIndx), _splitValue(c._splitValue),
    _leftChild(NULL), _rightChild(NULL), _scores(c._scores),
    _predictedClass(c._predictedClass), _maxDepth(c._maxDepth)
{
    // define properties
    declareProperty("maxDepth", new drwnRangeProperty(&_maxDepth, 1, DRWN_INT_MAX));

    // deep copy left and right children
    if (c._leftChild != NULL) {
        _leftChild = new drwnDecisionTree(*c._leftChild);
    }
    if (c._rightChild != NULL) {
        _rightChild = new drwnDecisionTree(*c._rightChild);
    }
}

drwnDecisionTree::~drwnDecisionTree()
{
    if (_leftChild != NULL) delete _leftChild;
    if (_rightChild != NULL) delete _rightChild;
}

// initialization
void drwnDecisionTree::initialize(unsigned n, unsigned k)
{
    drwnClassifier::initialize(n, k);
    if (_leftChild != NULL) delete _leftChild;
    if (_rightChild != NULL) delete _rightChild;
    _leftChild = _rightChild = NULL;
    _scores = VectorXd::Constant(k, log(1.0 / (double)k));
    _predictedClass = 0;
}

// i/o
bool drwnDecisionTree::save(drwnXMLNode& xml) const
{
    drwnClassifier::save(xml);

    drwnAddXMLAttribute(xml, "splitIndx", toString(_splitIndx).c_str(), false);
    drwnAddXMLAttribute(xml, "splitValue", toString(_splitValue).c_str(), false);
    drwnXMLUtils::serialize(xml, _scores);

    if (_leftChild != NULL) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnDecisionTree", NULL, false);
        _leftChild->save(*node);
    }

    if (_rightChild != NULL) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnDecisionTree", NULL, false);
        _rightChild->save(*node);
    }

    return true;
}

bool drwnDecisionTree::load(drwnXMLNode& xml)
{
    drwnClassifier::load(xml);

    if (_leftChild != NULL) delete _leftChild;
    if (_rightChild != NULL) delete _rightChild;

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "splitIndx") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "splitValue") != NULL);

    _splitIndx = atoi(drwnGetXMLAttribute(xml, "splitIndx"));
    _splitValue = atof(drwnGetXMLAttribute(xml, "splitValue"));
    drwnXMLUtils::deserialize(xml, _scores);
    if (_scores.size() == 0) {
        _predictedClass = -1;
    } else {
        _scores.maxCoeff(&_predictedClass);
    }

    // create and load children
    if (_splitIndx != -1) {
        _leftChild = new drwnDecisionTree();
        drwnXMLNode *childNode = xml.first_node("drwnDecisionTree");
        DRWN_ASSERT(childNode != NULL);
        _leftChild->load(*childNode);
        _rightChild = new drwnDecisionTree();
        childNode = childNode->next_sibling("drwnDecisionTree");
        DRWN_ASSERT(childNode != NULL);
        _rightChild->load(*childNode);
        DRWN_ASSERT(childNode->next_sibling("drwnDecisionTree") == NULL);
    } else {
        DRWN_ASSERT(xml.first_node("drwnDecisionTree") == NULL);
        _leftChild = NULL;
        _rightChild = NULL;
    }

    return true;
}

// training
double drwnDecisionTree::train(const drwnClassifierDataset& dataset)
{
    if (dataset.hasWeights()) {
        return train(dataset.features, dataset.targets, dataset.weights);
    } else {
        return train(dataset.features, dataset.targets);
    }
}

double drwnDecisionTree::train(const vector<vector<double> >& features,
    const vector<int>& targets)
{
    vector<double> weights(targets.size(), 1.0);
    return train(features, targets, weights);
}

double drwnDecisionTree::train(const vector<vector<double> >& features,
    const vector<int>& targets, const vector<double>& weights)
{
    // sort feature values
    vector<vector<int> > sortIndex;
    if (CACHE_SORTED_INDEXES) {
        computeSortedFeatureIndex(features, sortIndex);
    }

    // mark valid training samples
    drwnBitArray sampleIndex(features.size());
    sampleIndex.ones();
    for (int i = 0; i < sampleIndex.size(); i++) {
        if (targets[i] < 0) sampleIndex.clear(i);
    }

    // learn the decision tree
    learnDecisionTree(features, targets, weights, sortIndex, sampleIndex);

    // TODO: compute some score
    return 0.0;
}

// evaluation (log-probability)
void drwnDecisionTree::getClassScores(const vector<double>& features,
    vector<double>& outputScores) const
{
    DRWN_ASSERT((int)features.size() == _nFeatures);

#if 0
    if (_splitIndx < 0) {
        outputScores.resize(_scores.size());
        Eigen::Map<VectorXd>(&outputScores[0], outputScores.size()) = _scores;
        return;
    }

    if (features[_splitIndx] < _splitValue) {
        DRWN_ASSERT(_leftChild != NULL);
        _leftChild->getClassScores(features, outputScores);
    } else {
        DRWN_ASSERT(_rightChild != NULL);
        _rightChild->getClassScores(features, outputScores);
    }
#else
    // fast version (eliminate function calls)
    const drwnDecisionTree *p = this;
    while (p->_splitIndx >= 0) {
        p = (features[p->_splitIndx] < p->_splitValue) ? p->_leftChild : p->_rightChild;
    }

    outputScores.resize(p->_scores.size());
    Eigen::Map<VectorXd>(&outputScores[0], outputScores.size()) = p->_scores;
#endif
}


// evaluation (classification)
int drwnDecisionTree::getClassification(const vector<double>& features) const
{
    DRWN_ASSERT((int)features.size() == _nFeatures);
#if 0
    if (_splitIndx < 0) {
        return _predictedClass;
    } else if (features[_splitIndx] < _splitValue) {
        DRWN_ASSERT(_leftChild != NULL);
        return _leftChild->getClassification(features);
    } else {
        DRWN_ASSERT(_rightChild != NULL);
        return _rightChild->getClassification(features);
    }
#else
    // fast version (eliminate function calls)
    const drwnDecisionTree *p = this;
    while (p->_splitIndx >= 0) {
        p = (features[p->_splitIndx] < p->_splitValue) ? p->_leftChild : p->_rightChild;
    }

    return p->_predictedClass;
#endif
}

// training
void drwnDecisionTree::computeSortedFeatureIndex(const vector<vector<double> >& x,
    const drwnBitArray& sampleIndex, int featureIndx, vector<int>& featureSortIndex)
{
    // determine number of samples and features
    DRWN_ASSERT(sampleIndex.empty() || (sampleIndex.size() == (int)x.size()));
    const int numSamples = sampleIndex.empty() ? (int)x.size() : sampleIndex.count();
    if (numSamples == 0) {
        featureSortIndex.clear();
        return;
    }

    const int numFeatures = (int)x[0].size();
    DRWN_ASSERT((featureIndx >= 0) && (featureIndx < numFeatures));

    // sort feature values
    vector<pair<double, int> > sortedFeatures(numSamples);
    if (sampleIndex.empty()) {
        for (int i = 0; i < (int)x.size(); i++) {
            sortedFeatures[i].first = x[i][featureIndx];
            sortedFeatures[i].second = i;
        }
    } else {
        for (int i = 0, j = 0; i < (int)x.size(); i++) {
            if (sampleIndex[i]) {
                sortedFeatures[j].first = x[i][featureIndx];
                sortedFeatures[j].second = i;
                j += 1;
            }
        }
    }
    sort(sortedFeatures.begin(), sortedFeatures.end());

    featureSortIndex.resize(numSamples);
    for (int i = 0; i < numSamples; i++) {
        featureSortIndex[i] = sortedFeatures[i].second;
    }
}

void drwnDecisionTree::computeSortedFeatureIndex(const vector<vector<double> >& x,
    vector<vector<int> >& sortIndex)
{
    DRWN_FCN_TIC;
    DRWN_LOG_VERBOSE("pre-computing sorted feature indices for decision tree learning...");

    // determine number of samples and features
    const int numSamples = (int)x.size();
    if (numSamples == 0) {
        sortIndex.clear();
        return;
    }

    const int numFeatures = (int)x[0].size();

    // sort feature values
    sortIndex.resize(numFeatures);
    for (int i = 0; i < numFeatures; i++) {
        computeSortedFeatureIndex(x, drwnBitArray(), i, sortIndex[i]);
    }

    DRWN_FCN_TOC;
}

void drwnDecisionTree::learnDecisionTree(const vector<vector<double> >& x,
    const vector<int>& y, const vector<double>& w,
    const vector<vector<int> >& sortIndex, const drwnBitArray& sampleIndex)
{
    DRWN_ASSERT_MSG((x.size() > 0) && (y.size() == x.size()) && (w.size() == x.size()),
        x.size() << " > " << 0 << " && " << y.size() << " == " << x.size()
        << " && " << w.size() << " == " << x.size());

    const int numSamples = (int)x.size();

    DRWN_ASSERT(sortIndex.empty() || (sortIndex.size() == (unsigned)_nFeatures));
    DRWN_LOG_DEBUG("learning a " << _nClasses << "-class decision tree to depth "
        << _maxDepth << " with " << sampleIndex.count() << " samples (from "
        << numSamples << ") of length " << _nFeatures);

    // compute weighted class counts for this branch
    vector<double> classCounts(_nClasses, 0.0);
    double totalWeight = 0.0;
    for (int i = 0; i < numSamples; i++) {
        if (!sampleIndex[i]) continue;
        DRWN_ASSERT((y[i] >= 0) && (y[i] < _nClasses) && (w[i] >= 0.0));
        classCounts[y[i]] += w[i];
        totalWeight += w[i];
    }

    double H;
    switch (SPLIT_CRITERION) {
    case DRWN_DT_SPLIT_ENTROPY:
        H = totalWeight * drwn::entropy(classCounts);
        break;
    case DRWN_DT_SPLIT_MISCLASS:
        H = totalWeight - drwn::maxElem(classCounts);
        break;
    case DRWN_DT_SPLIT_GINI:
        H = totalWeight * drwn::gini(classCounts);
        break;
    default:
        H = 0.0;
        DRWN_ASSERT(false);
    }

    // leaf node so simply count classes and return
    if ((_maxDepth <= 0) || (H < DRWN_EPSILON)) {
        _scores = ((Eigen::Map<VectorXd>(&classCounts[0], _nClasses).array() + DRWN_EPSILON) /
            (totalWeight + _nClasses * DRWN_EPSILON)).array().log();
        _scores.maxCoeff(&_predictedClass);
        _bValid = true;
        DRWN_LOG_DEBUG("leaf distribution: " << toString(classCounts) << " (H = " << H << ")");
        return;
    }

    if (numSamples > MAX_FEATURE_THRESHOLDS) {
        DRWN_LOG_WARNING_ONCE("only trying " << MAX_FEATURE_THRESHOLDS << " threshold values");
    }

    const unsigned nThreads = std::min((unsigned)_nFeatures,
        std::max((unsigned)1, drwnThreadPool::MAX_THREADS));
    // define feature sets
    vector<set<int> > featureSets(nThreads);
    for (int i = 0; i < _nFeatures; i++) {
        featureSets[i % nThreads].insert(i);
    }

    // create thread pool
    drwnThreadPool threadPool;
    vector<drwnDecisionTreeThread *> decisionTreeJobs(nThreads,
        (drwnDecisionTreeThread *)NULL);
    threadPool.start();
    for (unsigned i = 0; i < nThreads; i++) {
        // create and add job
        decisionTreeJobs[i] = new drwnDecisionTreeThread(featureSets[i],
            x, y, w, sortIndex, sampleIndex, classCounts, H, totalWeight);
        threadPool.addJob(decisionTreeJobs[i]);
    }
    threadPool.finish();

    int bestFeature = -1;
    double bestScore = DRWN_EPSILON;
    double bestSplit = 0.0;

    for (unsigned i = 0; i < nThreads; i++) {
        if (decisionTreeJobs[i]->bestScore > bestScore) {
            bestFeature = decisionTreeJobs[i]->bestFeature;
            bestScore = decisionTreeJobs[i]->bestScore;
            bestSplit = decisionTreeJobs[i]->bestSplit;
        }

        delete decisionTreeJobs[i];
    }

    // create tree node
    if (bestFeature == -1) {
        DRWN_LOG_WARNING("could not find feature to split at " << _maxDepth);
        _scores = ((Eigen::Map<VectorXd>(&classCounts[0], _nClasses).array() + DRWN_EPSILON) /
            (totalWeight + _nClasses * DRWN_EPSILON)).array().log();
        _scores.maxCoeff(&_predictedClass);
        _bValid = true;
        DRWN_LOG_DEBUG("leaf distribution: " << toString(classCounts));
        return;
    }

    DRWN_LOG_DEBUG("best split on var " << bestFeature << " at " << bestSplit
        << " with score " << bestScore);
    _splitIndx = bestFeature;
    _splitValue = bestSplit;

    // learn left tree
    drwnBitArray childSampleIndex(numSamples);
    for (int i = 0; i < numSamples; i++) {
        if (sampleIndex[i]) {
            if ((x[i][bestFeature] < bestSplit) || ((LEAKAGE > 0.0) && (drand48() < LEAKAGE))) {
                childSampleIndex.set(i);
            }
        }
    }
    _leftChild = new drwnDecisionTree(_nFeatures, _nClasses);
    _leftChild->_maxDepth = _maxDepth - 1;
    _leftChild->learnDecisionTree(x, y, w, sortIndex, childSampleIndex);

    // learn right tree
    if (LEAKAGE == 0.0) {
        childSampleIndex.negate();
        childSampleIndex.bitwiseand(sampleIndex);
    } else {
        childSampleIndex.zeros();
        for (int i = 0; i < numSamples; i++) {
            if (sampleIndex[i]) {
                if ((x[i][bestFeature] >= bestSplit) || (drand48() < LEAKAGE)) {
                    childSampleIndex.set(i);
                }
            }
        }
    }
    _rightChild = new drwnDecisionTree(_nFeatures, _nClasses);
    _rightChild->_maxDepth = _maxDepth - 1;
    _rightChild->learnDecisionTree(x, y, w, sortIndex, childSampleIndex);

    _bValid = true;
}

// drwnDecisionTreeConfig ---------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnDecisionTree
//! \b maxDepth :: maximum depth of each decision tree (default: 2)\n
//! \b maxThresholds :: maximum number of thresholds to try during learning (default: 1000)\n
//! \b minSamples :: minimum number of samples after first split (default: 10)\n
//! \b leakage :: probability that a training sample leaks to both sides of a split (default: 0.0)\n
//! \b split :: split criterion for learning (ENTROPY (default), MISCLASS, GINI)\n
//! \b cacheSortIndex :: pre-cache sorted feature indexes for faster learning (default: true)\n

class drwnDecisionTreeConfig : public drwnConfigurableModule {
public:
    drwnDecisionTreeConfig() : drwnConfigurableModule("drwnDecisionTree") { }
    ~drwnDecisionTreeConfig() { }

    void usage(ostream &os) const {
        os << "      maxDepth      :: maximum depth (default: "
           << drwnDecisionTree::MAX_DEPTH << ")\n";
        os << "      maxThresholds :: maximum number of thresholds to try during learning (default: "
           << drwnDecisionTree::MAX_FEATURE_THRESHOLDS << ")\n";
        os << "      minSamples    :: minimum samples after first split (default: "
           << drwnDecisionTree::MIN_SAMPLES << ")\n";
        os << "      leakage       :: probability that a training sample leaks to both sides of a split (default: "
           << drwnDecisionTree::LEAKAGE << ")\n";
        os << "      split         :: split criterion for learning (ENTROPY (default), MISCLASS, GINI)\n";
        os << "      cacheSortIndex:: pre-cache sorted feature indexes for faster learning (default: "
           << (drwnDecisionTree::CACHE_SORTED_INDEXES ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxDepth")) {
            drwnDecisionTree::MAX_DEPTH = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxThresholds")) {
            drwnDecisionTree::MAX_FEATURE_THRESHOLDS = std::max(1, atoi(value));
        } else if (!strcmp(name, "minSamples")) {
            drwnDecisionTree::MIN_SAMPLES = std::max(0, atoi(value));
        } else if (!strcmp(name, "leakage")) {
            drwnDecisionTree::LEAKAGE = std::min(std::max(0.0, atof(value)), 1.0);
        } else if (!strcmp(name, "split")) {
            if (!strcasecmp(value, "ENTROPY")) {
                drwnDecisionTree::SPLIT_CRITERION = DRWN_DT_SPLIT_ENTROPY;
            } else if (!strcasecmp(value, "MISCLASS")) {
                drwnDecisionTree::SPLIT_CRITERION = DRWN_DT_SPLIT_MISCLASS;
            } else if (!strcasecmp(value, "GINI")) {
                drwnDecisionTree::SPLIT_CRITERION = DRWN_DT_SPLIT_GINI;
            } else {
                DRWN_LOG_FATAL("unrecognized configuration value " << value
                    << " for option " << name << " in " << this->name());
            }
        } else if (!strcmp(name, "cacheSortIndex")) {
            drwnDecisionTree::CACHE_SORTED_INDEXES = drwn::trueString(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnDecisionTreeConfig gDecisionTreeConfig;

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelSegModel.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

#include "drwnPixelSegModel.h"

using namespace std;

// threading for cross-validating weights ---------------------------------

class GridSearchJob : public drwnThreadJob {
protected:
    const drwnPixelSegModel &_model;
    const string _baseName;
    const vector<double> _pairwiseContrastValues;
    const vector<double> _robustPottsValues;

    map<pair<double, double>, double>& _scores;

public:
    GridSearchJob(const drwnPixelSegModel& model, const string& baseName,
        const vector<double>& pairwiseContrastValues, const vector<double>& robustPottsValues,
        map<pair<double, double>, double>& scores) :
        _model(model), _baseName(baseName), _pairwiseContrastValues(pairwiseContrastValues),
        _robustPottsValues(robustPottsValues), _scores(scores) { /* do nothing */ }
    ~GridSearchJob() { /* do nothing */ }

    void operator()() {
        DRWN_ASSERT(!_pairwiseContrastValues.empty() && !_robustPottsValues.empty());

        lock();
        DRWN_LOG_VERBOSE("evaluating model weights on " << _baseName << "...");
        unlock();

        const string imgFilename = gMultiSegConfig.filename("imgDir", _baseName, "imgExt");
        const string lblFilename = gMultiSegConfig.filename("lblDir", _baseName, "lblExt");
        const string segFilename = gMultiSegConfig.filename("segDir", _baseName, "segExt");

        // load image and cache unary potentials
        drwnSegImageInstance instance(imgFilename.c_str());
        _model.cacheUnaryPotentials(&instance);

        // load superpixels
        if ((_robustPottsValues.size() > 1) || (_robustPottsValues[0] != 0.0)) {
            if (!drwnFileExists(segFilename.c_str())) {
                lock();
                DRWN_LOG_WARNING("superpixel file \"" << segFilename << "\" does not exist");
                unlock();
            } else {
                ifstream ifs(segFilename.c_str(), ios::in | ios::binary);
                instance.superpixels.read(ifs);
                ifs.close();
            }
        }

        // load labels
        MatrixXi labels(instance.height(), instance.width());
        drwnLoadPixelLabels(labels, lblFilename.c_str(), _model.numLabels());

        const double numUnknown = (double)(labels.array() == -1).cast<int>().sum();

        // predict labels
        for (unsigned i = 0; i < _robustPottsValues.size(); i++) {
            for (unsigned j = 0; j < _pairwiseContrastValues.size(); j++) {

                drwnRobustPottsCRFInference inf;
                inf.alphaExpansion(&instance, _pairwiseContrastValues[j], _robustPottsValues[i]);

                // unweighted error
                double errors = (double)(instance.pixelLabels.array() != labels.array()).cast<int>().sum() - numUnknown;

                // update scores
                const pair<double, double> w(_pairwiseContrastValues[j], _robustPottsValues[i]);
                lock();
                if (_scores.find(w) == _scores.end()) {
                    _scores[w] = errors;
                } else {
                    _scores[w] += errors;
                }
                unlock();
            }
        }
    }
};

// drwnPixelSegModel class ------------------------------------------------

drwnPixelSegModel::drwnPixelSegModel() : _pixelContrastWeight(0.0), _robustPottsWeight(0.0)
{
    // default feature generator
    _featureGenerator = new drwnSegImageStdPixelFeatures();
}

drwnPixelSegModel::drwnPixelSegModel(const drwnPixelSegModel& model) :
    _featureGenerator(model._featureGenerator->clone()),
    _classTrainingWeights(model._classTrainingWeights),
    _pixelFeatureWhitener(model._pixelFeatureWhitener),
    _pixelUnaryModel(model._pixelUnaryModel),
    _pixelContrastWeight(model._pixelContrastWeight),
    _robustPottsWeight(model._robustPottsWeight)
{
    _pixelClassModels.reserve(model._pixelClassModels.size());
    for (size_t i = 0; i < model._pixelClassModels.size(); i++) {
        _pixelClassModels.push_back(model._pixelClassModels[i]->clone());
    }
}

drwnPixelSegModel::~drwnPixelSegModel() {
    clear();
    delete _featureGenerator;
}

// i/o
void drwnPixelSegModel::clear()
{
    _classTrainingWeights.clear();
    for (unsigned i = 0; i < _pixelClassModels.size(); i++) {
        delete _pixelClassModels[i];
    }
    _pixelClassModels.clear();
}

bool drwnPixelSegModel::save(drwnXMLNode& xml) const
{
    // save class training weights
    if (!_classTrainingWeights.empty()) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "classTrainingWeights", NULL, false);
        drwnXMLUtils::serialize(*node, (const char *)&_classTrainingWeights[0],
            sizeof(double) * _classTrainingWeights.size());
    }

    // save boosted classifiers
    for (unsigned i = 0; i < _pixelClassModels.size(); i++) {
        DRWN_ASSERT(_pixelClassModels[i] != NULL);
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "pixelClassModel", NULL, false);
        _pixelClassModels[i]->save(*node);
    }

    // save feature whitener and unary model
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "featureWhitener", NULL, false);
    _pixelFeatureWhitener.save(*node);
    node = drwnAddXMLChildNode(xml, "unaryModel", NULL, false);
    _pixelUnaryModel.save(*node);

    // save pairwise and robust potts weight
    drwnAddXMLAttribute(xml, "contrastWeight", toString(_pixelContrastWeight).c_str(), false);
    drwnAddXMLAttribute(xml, "robustPottsWeight", toString(_robustPottsWeight).c_str(), false);

    return true;
}

bool drwnPixelSegModel::load(drwnXMLNode& xml)
{
    clear();

    // load class training weight
    if (drwnCountXMLChildren(xml, "classTrainingWeights") != 0) {
        const int numClasses = gMultiSegRegionDefs.maxKey() + 1;
        drwnXMLNode *node = xml.first_node("classTrainingWeights");
        _classTrainingWeights.resize(numClasses);
        drwnXMLUtils::deserialize(*node, (char *)&_classTrainingWeights[0],
            sizeof(double) * _classTrainingWeights.size());
    }

    // load boosted classifiers
    for (drwnXMLNode *node = xml.first_node("pixelClassModel"); node != NULL; node = node->next_sibling("pixelClassModel")) {
        _pixelClassModels.push_back(new drwnBoostedClassifier());
        _pixelClassModels.back()->load(*node);
    }

    // load feature whitener and unary model
    drwnXMLNode *node = xml.first_node("featureWhitener");
    if (node == NULL) {
        DRWN_LOG_WARNING("XML is missing the featureWhitener");
    } else {
        _pixelFeatureWhitener.load(*node);
    }

    node = xml.first_node("unaryModel");
    if (node == NULL) {
        DRWN_LOG_WARNING("XML is missing the unaryModel");
    } else {
        _pixelUnaryModel.load(*node);
    }

    // load pairwise weight
    if (drwnGetXMLAttribute(xml, "contrastWeight") == NULL) {
        DRWN_LOG_WARNING("XML is missing the contrastWeight");
    } else {
        _pixelContrastWeight = atof(drwnGetXMLAttribute(xml, "contrastWeight"));
    }

    // load robust potts weight
    if (drwnGetXMLAttribute(xml, "robustPottsWeight") == NULL) {
        DRWN_LOG_WARNING("XML is missing the robustPottsWeight");
    } else {
        _robustPottsWeight = atof(drwnGetXMLAttribute(xml, "robustPottsWeight"));
    }

    return true;
}

// feature generation
void drwnPixelSegModel::setFeatureGenerator(const drwnSegImagePixelFeatures& featureGenerator)
{
    delete _featureGenerator;
    _featureGenerator = featureGenerator.clone();
}

// learning
void drwnPixelSegModel::learnTrainingClassWeights(const vector<string>& baseNames)
{
    // accumulate training data
    const int numClasses = gMultiSegRegionDefs.maxKey() + 1;
    _classTrainingWeights.resize(numClasses);

    // laplacian smoothing
    fill(_classTrainingWeights.begin(), _classTrainingWeights.end(), 1.0);

    // count class occurances
    for (int i = 0; i < (int)baseNames.size(); i++) {
        string lblFilename = gMultiSegConfig.filename("lblDir", baseNames[i], "lblExt");

        // accumulate "known" labels for this instance
        MatrixXi labels;
        drwnLoadPixelLabels(labels, lblFilename.c_str(), numClasses);
        vector<int> counts(numClasses, 0);
        for (int y = 0; y < labels.rows(); y++) {
            for (int x = 0; x < labels.cols(); x++) {
                if (labels(y, x) < 0) continue;
                counts[labels(y, x)] += 1;
            }
        }

        // accumulate labels for dataset
        for (int c = 0; c < numClasses; c++) {
            _classTrainingWeights[c] += (double)counts[c];
        }
    }

#if 0
    // invert and normalize class occurances to get weights
    double total = 0.0;
    for (int c = 0; c < numClasses; c++) {
        _classTrainingWeights[c] = 1.0 / _classTrainingWeights[c];
        total += _classTrainingWeights[c];
    }
#else
    // normalize class occurances to get weights
    double total = 0.0;
    for (int c = 0; c < numClasses; c++) {
        total += _classTrainingWeights[c];
    }
#endif

    Eigen::Map<VectorXd>(&_classTrainingWeights[0], _classTrainingWeights.size()) /= total;
}

void drwnPixelSegModel::learnBoostedPixelModels(const vector<string>& baseNames, int subSample)
{
    // accumulate training data
    const int numClasses = gMultiSegRegionDefs.maxKey() + 1;
    vector<vector<double> > featureVectors;
    vector<int> featureLabels;
    buildSampledTrainingSet(baseNames, "lblExt", numClasses,
        featureVectors, featureLabels, subSample, true);
    DRWN_ASSERT(!featureVectors.empty());

    // compute feature weights
    vector<double> featureWeights(featureVectors.size(), 1.0);
    vector<int> classCounts(numClasses, 0);
    for (unsigned i = 0; i < featureLabels.size(); i++) {
        classCounts[featureLabels[i]] += 1;
    }

    // allocate and classifiers
    _pixelClassModels.resize(numClasses, NULL);
    for (int i = 0; i < (int)_pixelClassModels.size(); i++) {
        DRWN_LOG_MESSAGE("training boosted classifier for class " << i << "...");
        DRWN_ASSERT(classCounts[i] > 0);

        // labels and weights for this classId
        vector<int> localLabels(featureLabels.size(), -1);
        const double w1 = 1.0 / (double)classCounts[i];
        const double w0 = 1.0 / (double)(featureLabels.size() - classCounts[i]);
        for (unsigned j = 0; j < featureLabels.size(); j++) {
            localLabels[j] = (featureLabels[j] == i) ? 1 : 0;
            featureWeights[j] = (featureLabels[j] == i) ? w1 : w0;
        }

        // train classifier
        if (_pixelClassModels[i] == NULL) {
            _pixelClassModels[i] = new drwnBoostedClassifier(featureVectors[0].size(), 2);
        } else {
            _pixelClassModels[i]->initialize(featureVectors[0].size(), 2);
        }
        double J = ((drwnClassifier *)_pixelClassModels[i])->train(featureVectors,
            localLabels, featureWeights);
        DRWN_LOG_VERBOSE("...training objective: " << J);
    }
}

void drwnPixelSegModel::learnPixelUnaryModel(const vector<string>& baseNames, int subSample)
{
    if (_pixelClassModels.empty()) {
        DRWN_LOG_WARNING("learning unary potentials from raw (not boosted) features");
    }

    // accumulate training data
    const int numClasses = gMultiSegRegionDefs.maxKey() + 1;
    drwnClassifierDataset dataset;
    buildSampledTrainingSet(baseNames, "lblExt", numClasses,
        dataset.features, dataset.targets, subSample, _pixelClassModels.empty());
    DRWN_ASSERT(!dataset.empty());

    if (!_classTrainingWeights.empty()) {
        dataset.weights.resize(dataset.targets.size(), 1.0);
	for (unsigned i = 0; i < dataset.targets.size(); i++) {
  	  if ((dataset.targets[i] >= 0) &&
	      (dataset.targets[i] < (int)_classTrainingWeights.size())) {
  	      dataset.weights[i] = _classTrainingWeights[dataset.targets[i]];
	    }
	}
    }

    // train feature whitener
    DRWN_LOG_VERBOSE("whitening feature vectors...");
    _pixelFeatureWhitener.train(dataset.features);
    _pixelFeatureWhitener.transform(dataset.features);

    // train multi-class logistic model
    DRWN_LOG_VERBOSE("learning multi-class logistic...");
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("trainPixelUnaryModel"));
    _pixelUnaryModel.initialize(dataset.numFeatures(), numClasses);
    _pixelUnaryModel.train(dataset);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("trainPixelUnaryModel"));

    // evaluate (on training data)
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        vector<int> predictedLabels;
        _pixelUnaryModel.getClassifications(dataset.features, predictedLabels);

        drwnConfusionMatrix confusion(numClasses);
        confusion.accumulate(dataset.targets, predictedLabels);
        confusion.write(cout);
        DRWN_LOG_VERBOSE("Accuracy: " << confusion.accuracy());
    }
}

void drwnPixelSegModel::learnPixelContrastWeight(const vector<string>& baseNames)
{
    // determine search space
    vector<double> pixelContrastValues = drwn::logSpaceVector(1.0, 128.0, 15);
    vector<double> robustPottsValues(1, _robustPottsWeight);

    // cross-validate weight
    crossValidateWeights(baseNames, pixelContrastValues, robustPottsValues);
}

void drwnPixelSegModel::learnPixelContrastWeight(double weight)
{
    DRWN_LOG_VERBOSE("setting pixel contrast weight to " << weight);
    DRWN_ASSERT(weight >= 0.0);
    _pixelContrastWeight = weight;
}

void drwnPixelSegModel::learnRobustPottsWeight(const vector<string>& baseNames)
{
    // determine search space
    vector<double> pixelContrastValues(1, _pixelContrastWeight);
    vector<double> robustPottsValues = drwn::logSpaceVector(0.125, 4.0, 11);

    // cross-validate weight
    crossValidateWeights(baseNames, pixelContrastValues, robustPottsValues);
}

void drwnPixelSegModel::learnRobustPottsWeight(double weight)
{
    DRWN_LOG_VERBOSE("setting robust potts weight to " << weight);
    DRWN_ASSERT(weight >= 0.0);
    _robustPottsWeight = weight;
}

void drwnPixelSegModel::learnPixelContrastAndRobustPottsWeights(const vector<string>& baseNames)
{
    // process in two stages for faster search
    vector<double> pixelContrastValues(1, 0.0);
    vector<double> robustPottsValues = drwn::logSpaceVector(0.125, 4.0, 11);

    // cross-validate weight
    crossValidateWeights(baseNames, pixelContrastValues, robustPottsValues);

    // stage-two: search around stage-one optimum
    pixelContrastValues = drwn::logSpaceVector(1.0, 128.0, 15);
    robustPottsValues = drwn::logSpaceVector(0.5 * _robustPottsWeight, 2.0 * _robustPottsWeight, 5);

    // cross-validate weight
    crossValidateWeights(baseNames, pixelContrastValues, robustPottsValues);
}

// inference
void drwnPixelSegModel::cacheUnaryPotentials(drwnSegImageInstance *instance) const
{
    if (_pixelClassModels.empty()) {
        instance->unaries.clear();
        instance->unaries.reserve(instance->size());

        drwnSegImagePixelFeatures *featureGenerator(_featureGenerator->clone());
        featureGenerator->cacheInstanceData(*instance);
        featureGenerator->appendAllPixelFeatures(instance->unaries);
        delete featureGenerator;
    } else {
        cacheBoostedPixelResponses(*instance);
    }

    for (int i = 0; i < instance->size(); i++) {
        vector<double> f(instance->unaries[i]);
        _pixelFeatureWhitener.transform(f);
        _pixelUnaryModel.getClassMarginals(f, instance->unaries[i]);
        for (unsigned j = 0; j < instance->unaries[i].size(); j++) {
            instance->unaries[i][j] = -1.0 * log(instance->unaries[i][j] + DRWN_DBL_MIN);
        }
    }
}

double drwnPixelSegModel::inferPixelLabels(drwnSegImageInstance *instance) const
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(instance != NULL);

    // compute image features
    if (instance->unaries.empty() || instance->unaries[0].empty()) {
        cacheUnaryPotentials(instance);
    }

    // run inference
    drwnRobustPottsCRFInference inf;
    inf.alphaExpansion(instance, _pixelContrastWeight, _robustPottsWeight);

    DRWN_FCN_TOC;
    return energy(instance);
}

double drwnPixelSegModel::inferPixelLabels(const vector<string>& baseNames,
    vector<MatrixXi>& predictedLabels) const
{
    predictedLabels.resize(baseNames.size());

    double e = 0.0;
    for (unsigned i = 0; i < baseNames.size(); i++) {
        const string imgFilename = gMultiSegConfig.filename("imgDir", baseNames[i], "imgExt");
        drwnSegImageInstance instance(imgFilename.c_str());

        if (_robustPottsWeight != 0.0) {
            const string segFilename = gMultiSegConfig.filename("segDir", baseNames[i], "segExt");
            if (!drwnFileExists(segFilename.c_str())) {
                DRWN_LOG_WARNING("superpixel file \"" << segFilename << "\" does not exist");
            } else {
                ifstream ifs(segFilename.c_str(), ios::binary);
                instance.superpixels.read(ifs);
                ifs.close();
            }
        }

        e += inferPixelLabels(&instance);
        predictedLabels[i] = instance.pixelLabels;
    }

    return e;
}

double drwnPixelSegModel::energy(drwnSegImageInstance *instance) const
{
    DRWN_ASSERT(instance != NULL);

    // compute image features
    if (instance->unaries.empty() || instance->unaries[0].empty()) {
        cacheUnaryPotentials(instance);
    }

    // compute energy
    drwnRobustPottsCRFInference inf;
    return inf.energy(instance, _pixelContrastWeight, _robustPottsWeight);
}

// pixel features
void drwnPixelSegModel::computeBoostedResponses(const vector<double>& x, vector<double>& y) const
{
    y.resize(_pixelClassModels.size());

    vector<double> f(2);
    for (unsigned i = 0; i < _pixelClassModels.size(); i++) {
        _pixelClassModels[i]->getClassScores(x, f);
        y[i] = f[0] - f[1];
    }
}

void drwnPixelSegModel::cacheBoostedPixelResponses(drwnSegImageInstance &instance) const
{
    DRWN_FCN_TIC;

    const int nFeatures = (int)_pixelClassModels.size();
    instance.unaries.clear();
    instance.unaries.reserve(instance.size());
    bool bUseCache = gMultiSegConfig.getBoolProperty(gMultiSegConfig.findProperty("useCache")) &&
        !instance.name().empty();
    bool bCompressedCache = bUseCache && gMultiSegConfig.getBoolProperty(gMultiSegConfig.findProperty("compressedCache"));
    bool bReadCacheSuccess = true;

    // check if instance is in the cache
    //! \todo refactor using persistent storage class
    string cacheFile = gMultiSegConfig.filebase("cacheDir", instance.name()) + string(".boosted.bin");
    if (bUseCache) {
        if (drwnFileExists(cacheFile.c_str())) {
            drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("readPixelFeatureCache"));
            vector<double> featureVector(nFeatures);
            ifstream ifs(cacheFile.c_str(), ifstream::in | ifstream::binary);
            DRWN_ASSERT(!ifs.fail());

            if (bCompressedCache) {
                drwnCompressionBuffer buffer;
                bReadCacheSuccess = buffer.read(ifs);
                if (bReadCacheSuccess) {
                    DRWN_ASSERT(buffer.originalBytes() == instance.size() * nFeatures * sizeof(double));
                    double *data = new double[instance.size() * nFeatures];
                    buffer.decompress((unsigned char *)data);
                    for (int i = 0; i < instance.size(); i++) {
                        memcpy(&featureVector[0], &data[i * nFeatures], nFeatures * sizeof(double));
                        instance.unaries.push_back(featureVector);
                    }
                    delete[] data;
                }
            } else {
                for (int i = 0; i < instance.size(); i++) {
                    ifs.read((char *)&featureVector[0], nFeatures * sizeof(double));

                    for (int j = 0; j < nFeatures; j++) {
                        DRWN_ASSERT_MSG(isfinite(featureVector[j]), cacheFile);
                    }
                    if (ifs.fail()) {
                        bReadCacheSuccess = false;
                        DRWN_LOG_WARNING("load pixel feature cache failed for "
                            << cacheFile << ". Re-computing pixel features...");
                        break;
                    }

                    instance.unaries.push_back(featureVector);
                }
            }
            ifs.close();
            drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("readPixelFeatureCache"));

            // was cache read a success?
            if (bReadCacheSuccess == true) {
                DRWN_FCN_TOC;
                return;
            } else {
                instance.unaries.clear();
            }
        }
    }

    // not in cache (or cache not enabled), so compute
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("computePixelFeatures"));

    // need to clone feature generator for thread safety
    drwnSegImagePixelFeatures *featureGenerator(_featureGenerator->clone());
    featureGenerator->cacheInstanceData(instance);
    for (int y = 0; y < instance.height(); y++) {
        for (int x = 0; x < instance.width(); x++) {
            instance.unaries.push_back(vector<double>());
            vector<double> v;
            featureGenerator->appendPixelFeatures(x, y, v);
            computeBoostedResponses(v, instance.unaries.back());
        }
    }
    delete featureGenerator;
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("computePixelFeatures"));

    // write to cache
    if (bUseCache) {
        drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("writePixelFeatureCache"));
        ofstream ofs(cacheFile.c_str(), ofstream::out | ofstream::binary);
        DRWN_ASSERT(!ofs.fail());
        if (bCompressedCache) {
            drwnCompressionBuffer buffer;
            double *data = new double[instance.size() * nFeatures * sizeof(double)];
            for (int i = 0; i < instance.size(); i++) {
                memcpy(&data[i * nFeatures], &instance.unaries[i][0], nFeatures * sizeof(double));
            }
            buffer.compress((unsigned char *)data, instance.size() * nFeatures * sizeof(double));
            buffer.write(ofs);
            delete[] data;
        } else {
            for (unsigned i = 0; i < instance.unaries.size(); i++) {
                ofs.write((char *)&instance.unaries[i][0], nFeatures * sizeof(double));
            }
        }
        DRWN_ASSERT(!ofs.fail());
        ofs.close();
        drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("writePixelFeatureCache"));
    }

    DRWN_FCN_TOC;
}

// load training data
void drwnPixelSegModel::buildSampledTrainingSet(const vector<string>& baseNames, const char *labelExt,
    int nLabels, vector<vector<double> >& featureVectors, vector<int>& featureLabels,
    int subSample, bool bRawFeatures) const
{
    // accumulate training data by sampling randomly within a grid of size delta-by-delta
    const int delta = std::max<int>((int)sqrt((float)subSample), 1);
    vector<int> classCounts(nLabels, 0);

    DRWN_LOG_MESSAGE("Building sampled dataset of " << (bRawFeatures ? "raw" : "boosted")
        << " features for " << nLabels << " labels...");
    int hProcessImage = drwnCodeProfiler::getHandle("processImage");
    drwnSegImagePixelFeatures *featureGenerator(_featureGenerator->clone());

    for (int i = 0; i < (int)baseNames.size(); i++) {
        drwnCodeProfiler::tic(hProcessImage);
        string imgFilename = gMultiSegConfig.filename("imgDir", baseNames[i], "imgExt");
        string lblFilename = gMultiSegConfig.filename("lblDir", baseNames[i], labelExt);

        // load image
        drwnSegImageInstance instance(imgFilename.c_str());

        // load labels
        MatrixXi labels(instance.height(), instance.width());
        drwnLoadPixelLabels(labels, lblFilename.c_str(), nLabels);

        if ((labels.array() == -1).all()) {
            DRWN_LOG_WARNING("instance " << baseNames[i] << " is all unknown");
            continue;
        }

        // accumulate "known" labels
        featureGenerator->cacheInstanceData(instance);
        for (int v = 0; v < instance.height(); v += delta) {
            for (int u = 0; u < instance.width(); u += delta) {

                // randomly sample within grid
                const int y = std::min((int)(v + delta * drand48()), instance.height() - 1);
                const int x = std::min((int)(u + delta * drand48()), instance.width() - 1);
                if (labels(y, x) < 0) continue;

                // accumulate (raw or boosted features)
                vector<double> v;
                featureGenerator->appendPixelFeatures(x, y, v);
                if (bRawFeatures) {
                    featureVectors.push_back(v);
                } else {
                    featureVectors.push_back(vector<double>());
                    computeBoostedResponses(v, featureVectors.back());
                }
                featureLabels.push_back(labels(y, x));
                classCounts[labels(y, x)] += 1;
            }
        }
        featureGenerator->clearInstanceData();

        DRWN_LOG_VERBOSE(baseNames[i] << " (" << (i + 1) << " of "
            << baseNames.size() << "): " << featureVectors.size() << " ("
            << toString(classCounts) << ") samples of size "
            << (featureVectors.empty() ? 0 : featureVectors.front().size())
            << " accumulated");

        // reserve some additional space (once we have a good idea of requirements)
        if (i == (int)baseNames.size() / 4) {
            featureVectors.reserve((int)(5 * featureVectors.size()));
            featureLabels.reserve((int)(5 * featureVectors.size()));
        }

        drwnCodeProfiler::toc(hProcessImage);
    }

    delete featureGenerator;
}

void drwnPixelSegModel::crossValidateWeights(const vector<string>& baseNames,
    const vector<double>& pairwiseContrastValues, const vector<double>& robustPottsValues)
{
    // cross-validate pairwise contrast and robust potts weights
    map<pair<double, double>, double> scores;
    int delta = std::max(1, (int)baseNames.size() / 100);

    // start thread pool and add jobs
    drwnThreadPool threadPool;
    threadPool.start();
    vector<GridSearchJob *> jobs;
    for (int i = 0; i < (int)baseNames.size(); i += delta) {
        jobs.push_back(new GridSearchJob(*this, baseNames[i],
            pairwiseContrastValues, robustPottsValues, scores));
        threadPool.addJob(jobs.back());
    }

    // finish running jobs and delete them
    threadPool.finish();
    for (unsigned i = 0; i < jobs.size(); i++) {
        delete jobs[i];
    }

    // find best weight
    double bestScore = numeric_limits<double>::max();
    for (map<pair<double, double>, double>::const_iterator it = scores.begin(); it != scores.end(); it++) {
        DRWN_LOG_VERBOSE("CONTRAST/ROBUST POTTS WEIGHTS "
            << setw(5) << setprecision(3) << it->first.first << " "
            << setw(5) << setprecision(3) << it->first.second << "\t"
            << setprecision(5) << it->second);
        if (it->second < bestScore) {
            _pixelContrastWeight = it->first.first;
            _robustPottsWeight = it->first.second;
            bestScore = it->second;
        }
    }
}

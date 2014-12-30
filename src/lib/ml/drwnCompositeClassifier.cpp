/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnCompositeClassifier.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnML.h"

using namespace std;
using namespace Eigen;

// drwnCompositeClassifier static members ------------------------------------

string drwnCompositeClassifier::BASE_CLASSIFIER = string("drwnBoostedClassifier");
drwnCompositeClassifierMethod drwnCompositeClassifier::METHOD = DRWN_ONE_VS_ALL;

// drwnCompositeClassifier ---------------------------------------------------

drwnCompositeClassifier::drwnCompositeClassifier() :
    drwnClassifier(), _baseClassifier(BASE_CLASSIFIER), _method(METHOD)
{
    // do nothing
}

drwnCompositeClassifier::drwnCompositeClassifier(unsigned n, unsigned k) :
    drwnClassifier(n, k), _baseClassifier(BASE_CLASSIFIER), _method(METHOD)
{
    initialize(n, k);
}

drwnCompositeClassifier::drwnCompositeClassifier(const drwnCompositeClassifier &c) :
    drwnClassifier(c),  _baseClassifier(c._baseClassifier), _method(c._method),
    _featureWhitener(c._featureWhitener), _calibrationWeights(c._calibrationWeights)
{
    // deep copy binary classifiers
    _binaryClassifiers.resize(c._binaryClassifiers.size(), NULL);
    for (unsigned i = 0; i < c._binaryClassifiers.size(); i++) {
        _binaryClassifiers[i] = c._binaryClassifiers[i]->clone();
    }
}

drwnCompositeClassifier::~drwnCompositeClassifier()
{
    for (unsigned i = 0; i < _binaryClassifiers.size(); i++) {
        delete _binaryClassifiers[i];
    }
}

// initialization
void drwnCompositeClassifier::initialize(unsigned n, unsigned k)
{
    drwnClassifier::initialize(n, k);
    for (unsigned i = 0; i < _binaryClassifiers.size(); i++) {
        delete _binaryClassifiers[i];
    }
    _binaryClassifiers.clear();
    _featureWhitener.clear();
    _calibrationWeights.initialize(_method == DRWN_ONE_VS_ALL ? k : k * (k - 1) / 2, k);
}

// i/o
bool drwnCompositeClassifier::save(drwnXMLNode& xml) const
{
    drwnClassifier::save(xml);

    // save meta-parameters
    drwnAddXMLAttribute(xml, "baseClassifier", _baseClassifier.c_str(), false);
    drwnAddXMLAttribute(xml, "method", toString((int)_method).c_str(), false);

    // save binary classifiers
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "binaryClassifiers", NULL, false);
    for (unsigned i = 0; i < _binaryClassifiers.size(); i++) {
        drwnXMLNode *subnode = drwnAddXMLChildNode(*node, _binaryClassifiers[i]->type());
        _binaryClassifiers[i]->save(*subnode);
    }

    // save feature whitener and calibration weights
    node = drwnAddXMLChildNode(xml, "featureWhitener", NULL, false);
    _featureWhitener.save(*node);

    node = drwnAddXMLChildNode(xml, "calibrationWeights", NULL, false);
    _calibrationWeights.save(*node);

    return true;
}

bool drwnCompositeClassifier::load(drwnXMLNode& xml)
{
    drwnClassifier::load(xml);

    // load meta-parameters
    if (drwnGetXMLAttribute(xml, "baseClassifier") == NULL) {
        DRWN_LOG_WARNING("XML is missing baseClassifier attribute");
    } else {
        _baseClassifier = string(drwnGetXMLAttribute(xml, "baseClassifier"));
    }
    if (drwnGetXMLAttribute(xml, "method") == NULL) {
        DRWN_LOG_WARNING("XML is missing method attribute");
    } else {
        _method = (drwnCompositeClassifierMethod)atoi(drwnGetXMLAttribute(xml, "method"));
    }

    // load binary classifiers
    drwnXMLNode *node = xml.first_node("binaryClassifiers");
    for (drwnXMLNode *child = node->first_node(); child != NULL; child = child->next_sibling()) {
        _binaryClassifiers.push_back(drwnClassifierFactory::get().createFromXML(*child));
    }

    // load feature whitener and calibration weights
    node = xml.first_node("featureWhitener");
    _featureWhitener.load(*node);

    node = xml.first_node("calibrationWeights");
    _calibrationWeights.load(*node);

    return true;
}

// training
double drwnCompositeClassifier::train(const drwnClassifierDataset& dataset)
{
    DRWN_FCN_TIC;

    // compute label weights
    vector<int> classCounts(_nClasses, 0);
    int totalCount = 0;
    for (int i = 0; i < dataset.size(); i++) {
        if (dataset.targets[i] >= 0) {
            classCounts[dataset.targets[i]] += 1;
            totalCount += 1;
        }
    }

    // learn binary classifiers
    vector<double> weights(dataset.size(), 1.0);
    vector<int> targets(dataset.size(), -1);
    for (unsigned i = 0; i < _binaryClassifiers.size(); i++) {
        delete _binaryClassifiers[i];
    }
    _binaryClassifiers.clear();

    switch (_method) {
    case DRWN_ONE_VS_ALL:
        for (int k = 0; k < _nClasses; k++) {
            if (classCounts[k] == 0) continue;
            DRWN_LOG_VERBOSE("...training <class " << (k + 1) << ">-vs-all");
            const double w1 = 1.0 / (double)classCounts[k];
            const double w0 = 1.0 / (double)(totalCount - classCounts[k]);

            for (int i = 0; i < dataset.size(); i++) {
                targets[i] = (dataset.targets[i] == k ? 1 : 0);
                weights[i] = (dataset.targets[i] == k ? w1 : w0);
            }

            drwnClassifier *c = drwnClassifierFactory::get().create(_baseClassifier.c_str());
            c->initialize(_nFeatures, 2);
            c->train(dataset.features, targets, weights);

            _binaryClassifiers.push_back(c);
        }
        break;
    case DRWN_ONE_VS_ONE:
        for (int k = 0; k < _nClasses; k++) {
            if (classCounts[k] == 0) continue;
            for (int l = 0; l < k; l++) {
                if (classCounts[l] == 0) continue;
                DRWN_LOG_VERBOSE("...training <class " << (k + 1) << ">-vs-<" << " class " << (l + 1) << ">");

                const double w1 = 1.0 / (double)classCounts[k];
                const double w0 = 1.0 / (double)classCounts[l];

                for (int i = 0; i < dataset.size(); i++) {
                    if ((dataset.targets[i] != k) && (dataset.targets[i] != l)) {
                        targets[i] = -1;
                    } else {
                        targets[i] = (dataset.targets[i] == k ? 1 : 0);
                        weights[i] = (dataset.targets[i] == k ? w1 : w0);
                    }
                }

                drwnClassifier *c = drwnClassifierFactory::get().create(_baseClassifier.c_str());
                c->initialize(_nFeatures, 2);
                c->train(dataset.features, targets, weights);

                _binaryClassifiers.push_back(c);
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unknown method in drwnCompositeClassifier::train");
    }

    // clear some memory
    targets.clear();
    weights.clear();

    // learn calibration weights
    vector<double> f(2);
    vector<vector<double> > features(dataset.size(), vector<double>(_binaryClassifiers.size(), 0));
    for (int i = 0; i < dataset.size(); i++) {
        for (unsigned j = 0; j < _binaryClassifiers.size(); j++) {
            _binaryClassifiers[j]->getClassScores(dataset.features[i], f);
            features[i][j] = f[0] - f[1];
        }
    }

    DRWN_LOG_VERBOSE("whitening feature vectors...");
    _featureWhitener.train(features);
    _featureWhitener.transform(features);

    // train multi-class logistic model
    DRWN_LOG_VERBOSE("learning calibration weights...");
    _calibrationWeights.initialize(_binaryClassifiers.size(), _nClasses);
    if (dataset.hasWeights()) {
        _calibrationWeights.train(features, dataset.targets, dataset.weights);
    } else {
        _calibrationWeights.train(features, dataset.targets);
    }

    _bValid = true;
    DRWN_FCN_TOC;
    return 0.0;
}

// evaluation (log-probability)
void drwnCompositeClassifier::getClassScores(const vector<double>& features,
    vector<double>& outputScores) const
{
    DRWN_ASSERT_MSG((int)features.size() == _nFeatures, (int)features.size() << " != " << _nFeatures);

    // extract binary classifier outputs
    vector<double> x(_binaryClassifiers.size());
    vector<double> f(2);
    for (unsigned i = 0; i < _binaryClassifiers.size(); i++) {
        _binaryClassifiers[i]->getClassScores(features, f);
        x[i] = f[0] - f[1];
    }

    // whiten the features
    _featureWhitener.transform(x);

    // calibrate multi-class output
    _calibrationWeights.getClassScores(x, outputScores);
}

// drwnCompositeClassifierConfig --------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnCompositeClassifier
//! \b baseClassifier :: base binary classifier (default: drwnBoostedClassifier)\n
//! \b method :: composition method (0: one-vs-all, 1: one-vs-one)

class drwnCompositeClassifierConfig : public drwnConfigurableModule {
public:
    drwnCompositeClassifierConfig() : drwnConfigurableModule("drwnCompositeClassifier") { }
    ~drwnCompositeClassifierConfig() { }

    void usage(ostream &os) const {
        os << "      baseClassifier :: base binary classifier (default: drwnBoostedClassifier)\n";
        os << "      method        :: composition method (0: one-vs-all, 1: one-vs-one)\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "baseClassifier")) {
            drwnCompositeClassifier::BASE_CLASSIFIER = string(value);
        } else if (!strcmp(name, "method")) {
            drwnCompositeClassifier::METHOD = (drwnCompositeClassifierMethod)atoi(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnCompositeClassifierConfig gCompositeClassifierConfig;

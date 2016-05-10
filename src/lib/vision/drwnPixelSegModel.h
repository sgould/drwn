/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelSegModel.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "Eigen/Core"

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"

using namespace std;

// drwnPixelSegModel class ------------------------------------------------
//! Implements a pixel-level CRF model for multi-class image segmentation
//! (pixel labeling).
//!
//! The model supports a contrast-sensitive pairwise smoothness prior, long
//! range pairwise terms determined from patch matching, and a higher-order
//! robust generalized potts consistency prior. The pairwise smoothness prior
//! is defined over an 8-connected neighbourhood for each pixel. The robust potts
//! prior is implemented on the superpixels defined in each drwnSegImageInstance.
//!
//! \sa drwnSegImageInstance
//! \sa \ref drwnProjMultiSeg

class drwnPixelSegModel : public drwnWriteable {
 protected:
    //! pixel feature generator
    drwnSegImagePixelFeatures *_featureGenerator;

    //! class training weights (normalized to sum to one)
    vector<double> _classTrainingWeights;

    //! boosted pixel classifiers for generating pixel features
    //! (if boosted pixel classifiers are not learned then the
    //! multi-class logistic is run on the raw features)
    vector<drwnBoostedClassifier *> _pixelClassModels;

    //! feature whitener for unary model
    drwnFeatureWhitener _pixelFeatureWhitener;
    //! pixelwise unary model
    drwnTMultiClassLogistic<drwnBiasJointFeatureMap> _pixelUnaryModel;
    //drwnTMultiClassLogistic<drwnSquareJointFeatureMap> _pixelUnaryModel;
    //drwnTMultiClassLogistic<drwnQuadraticJointFeatureMap> _pixelUnaryModel;

    //! weight for pairwise constrast-dependent smoothness term
    double _pixelContrastWeight;

    //! weight for (long range) auxiliary edges
    double _longRangeEdgeWeight;
    //! radius for long range edge match
    unsigned _longRangeMatchRadius;
    //! percentage of long range edges to include
    double _longRangeEdgeThreshold;

    //! weight for robust potts consistency term
    double _robustPottsWeight;

 public:
    //! default constructor
    drwnPixelSegModel();
    //! copy constructor
    drwnPixelSegModel(const drwnPixelSegModel& model);
    virtual ~drwnPixelSegModel();

    // i/o
    const char *type() const { return "pixelSegModel"; }
    //! clear learned model parameters
    virtual void clear();
    virtual bool save(drwnXMLNode& xml) const;
    virtual bool load(drwnXMLNode& xml);

    //! return the number of labels that this model has been trained to recognize
    int numLabels() const { return _pixelUnaryModel.numClasses(); }

    // feature generation
    //! set the object used to generate pixel features from the image instance
    void setFeatureGenerator(const drwnSegImagePixelFeatures& featureGenerator);
    //! return the object used to generate pixel features from the image instance
    const drwnSegImagePixelFeatures& getFeatureGenerator() const {
        return *_featureGenerator;
    }

    // learning
    //! learn the training class weights by counting class occurrance
    void learnTrainingClassWeights(const vector<string>& baseNames);
    //! learn boosted classifiers for predicting classes from pixel features
    void learnBoostedPixelModels(const vector<string>& baseNames, int subSample = 0);
    //! learn unary potentials for calibrating pixel predictions
    void learnPixelUnaryModel(const vector<string>& baseNames, int subSample = 0);
    //! learn the weight of the pairwise constrast term
    void learnPixelContrastWeight(const vector<string>& baseNames);
    //! set the weight of the pairwise contrast term
    void learnPixelContrastWeight(double weight);
    //! learn the weight of the long range pairwise term
    void learnLongRangePairwiseWeight(const vector<string>& baseNames, double threshold, unsigned radius = 4);
    //! set the weight of the long range pairwise term
    void learnLongRangePairwiseWeight(double weight, double threshold, unsigned radius = 4);
    //! learn the weight of the robust potts consistency term
    void learnRobustPottsWeight(const vector<string>& baseNames);
    //! set the weight of the robust potts consistency term
    void learnRobustPottsWeight(double weight);
    //! learn the weights of the pairwise contrast and robust potts consistency terms jointly
    void learnPixelContrastAndRobustPottsWeights(const vector<string>& baseNames);
    //! learn the contrast and long-range weights jointly
    void learnPixelContrastAndLongRangeWeights(const vector<string>& baseNames, double threshold, unsigned radius = 4);

    //! get the weight of the pairwise contrast term
    double getPairwiseContrastWeight() const { return _pixelContrastWeight; }
    //! get the weight of the long range pairwise term
    double getLongRangePairwiseWeight() const { return _longRangeEdgeWeight; }
    //! get the weight of the robust potts term
    double getRobustPottsWeight() const { return _robustPottsWeight; }

    // inference
    //! cache the unary potentials inside the instance object
    void cacheUnaryPotentials(drwnSegImageInstance *instance) const;
    //! cache the long-range edges inside the instance object
    void cacheLongRangeEdges(drwnSegImageInstance *instance) const;
    //! infer the pixel labels using learned model parameters and store the predicted
    //! labels inside the instance object
    double inferPixelLabels(drwnSegImageInstance *instance) const;
    //! infer pixel labels for a set of images and return predicted labels in
    //! \p predictedLabels
    virtual double inferPixelLabels(const vector<string>& baseNames,
        vector<MatrixXi>& predictedLabels) const;
    //! return the energy of a given labeling according to the model parameters
    virtual double energy(drwnSegImageInstance *instance) const;

 protected:
    //! compute boosted pixel features \p y as a function of raw features \p x
    void computeBoostedResponses(const vector<double>& x, vector<double>& y) const;
    //! cache all boosted pixel features in \p instance
    void cacheBoostedPixelResponses(drwnSegImageInstance &instance) const;

    //! dataset creation for training
    void buildSampledTrainingSet(const vector<string>& baseNames, const char *labelExt,
        int nLabels, vector<vector<double> >& featureVectors, vector<int>& featureLabels,
        int subSample, bool bRawFeatures) const;

    //! cross-validate pairwise contrast and robust potts weights
    void crossValidateWeights(const vector<string>& baseNames,
        const vector<double>& pairwiseContrastValues,
        const vector<double>& robustPottsValues, const vector<double>& longRangeValues);
};


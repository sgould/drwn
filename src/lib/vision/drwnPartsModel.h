/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPartsModel.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;

// Forward Declarations -----------------------------------------------------

class drwnPartsModel;

// drwnPartsAssignment ------------------------------------------------------
//! Class for holding as assignment to part locations and occlusion variables.

class drwnPartsAssignment {
 public:
    const drwnPartsModel *model;  //!< model that generated this assignment
    cv::Point centroid;           //!< centroid location
    vector<cv::Point> locations;  //!< parts locations
    vector<bool> occluded;        //!< only includes parts
    double score;                 //!< score or multiplier (for constraints)

 public:
    drwnPartsAssignment();
    drwnPartsAssignment(unsigned numParts);
    drwnPartsAssignment(const drwnPartsAssignment& assignment);
    ~drwnPartsAssignment();

    int numParts() const { return (int)locations.size(); }
    int numEqual(const drwnPartsAssignment& assignment) const;

    unsigned size() const { return locations.size() + occluded.size() + 1; }
    bool empty() const { return locations.empty(); }
    void clear();
    void resize(unsigned numParts);

    void print(std::ostream& os = std::cout) const;

    drwnPartsAssignment& operator=(const drwnPartsAssignment& assignment);
    bool operator==(const drwnPartsAssignment& assignment) const;
    const cv::Point& operator[](unsigned indx) const { return locations[indx]; }
};

// drwnDeformationCost ------------------------------------------------------
//! Structure for holding dx, dy, dx^2 and dy^2 deformation costs.

class drwnDeformationCost : public drwnStdObjIface {
 public:
    double dx, dy, dx2, dy2;

 public:
    drwnDeformationCost() : dx(1.0), dy(1.0), dx2(0.0), dy2(0.0) { /* do nothing */ }
    drwnDeformationCost(double a, double b, double c, double d) :
        dx(a), dy(b), dx2(c), dy2(d) { /* do nothing */ }
    drwnDeformationCost(const drwnDeformationCost& c) :
        dx(c.dx), dy(c.dy), dx2(c.dx2), dy2(c.dy2) { /* do nothing */ }
    ~drwnDeformationCost() { /* do nothing */ }

    // i/o
    const char *type() const { return "drwnDeformationCost"; }
    drwnDeformationCost* clone() const { return new drwnDeformationCost(*this); }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    inline double cost(const cv::Point& p, const cv::Point& q) const {
        const double deltaX = fabs((double)(p.x - q.x));
        const double deltaY = fabs((double)(p.y - q.y));
        return ((dx + dx2 * deltaX) * deltaX + (dy + dy2 * deltaY) * deltaY);
    }
};

// drwnPartsInference -------------------------------------------------------
//! Helper class for running inference in a (constellation) parts-based model.
//! Supports linear and quadratic distance transforms for deformation costs.
//! Computes
//!    argmax_{x,c,z} \sum_i m_i(x_i, z_i) + \lambda \sum_i d_i(x_i, c) + p(c)
//! where
//!    m_i(x, 0) = matchingCost_i(x) + priorCost_i(x)
//!    m_i(x, 1) = occlusionCost + priorCost_i(x)
//!    d_i(x, c) = [dx, dy]^T fabs(x - o - c) + [dx2, dy2]^T (x - o - c).^2
//!

class drwnPartsInference {
 public:
    static int MESSAGE_PASSING_SCALE;
    static double DEFAULT_LAMBDA;
    static double DEFAULT_OCCLUSION_COST;

 protected:
    double _lambda;
    double _occlusionCost;

    int _width;
    int _height;
    int _nParts;

    cv::Mat _centroidPrior;
    vector<pair<cv::Mat, cv::Mat> > _partCosts;
    vector<cv::Point> _partOffsets;
    vector<drwnDeformationCost> _pairwiseCosts;

 public:
    drwnPartsInference(int numParts, int width, int height);
    ~drwnPartsInference();

    // define problem
    void setCentroidPrior(const cv::Mat& centroidPrior = cv::Mat());
    void setPartCosts(int partId, const cv::Mat& matchingCost,
        const cv::Point& offset, const drwnDeformationCost& pairwiseCost,
        const cv::Mat& occlusionCost = cv::Mat());

    // inference
    drwnPartsAssignment inference() const;
    drwnPartsAssignment inference(const cv::Point& centroid) const;

    // negative max marginals (for each centroid location)
    cv::Mat energyLandscape() const;

 protected:
    // inference message passing between location variables
    cv::Mat computeLocationMessage(const cv::Mat& belief, const drwnDeformationCost& dcost,
        const cv::Mat& msgIn = cv::Mat()) const;
};

// drwnPart -----------------------------------------------------------------
//! A part is defined as a template over a number of channels (possibly one),
//! an offset from the object centroid, and a deformation cost. The part is
//! scored as w^T F(x) where w are the template weights, and F(x) is the feature
//! vector at location x in feature space. Note for edge templates this can be
//! in pixels, but generally will be in some multiple of pixels. It is up to
//! the inference model to handle conversion from feature space to pixel space.

class drwnPart : public drwnStdObjIface {
 public:
    static int MATCH_MODE;

 protected:
    cv::Size _extent;           //!< width and height of template (in pixels)
    vector<cv::Mat> _weights;   //!< template coefficients (in feature space)
    cv::Point _offset;          //!< centroid offset (in pixels)
    drwnDeformationCost _dcost; //!< deformation cost (dx, dy, dx^2, dy^2) (in pixels)

 public:
    drwnPart();
    drwnPart(const cv::Size& extent, unsigned channels = 1);
    drwnPart(const drwnPart& part);
    virtual ~drwnPart();

    // parameters
    int channels() const { return (int)_weights.size(); }
    int width() const { return _extent.width; }
    int height() const { return _extent.height; }

    // i/o
    const char *type() const { return "drwnPart"; }
    drwnPart *clone() const { return new drwnPart(*this); }
    void clear();
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);
    void swap(drwnPart& part);

    // learning (single and multi-channel variants)
    void setExtent(const cv::Size& extent) { _extent = extent; }
    void setExtent(int width, int height) { _extent = cv::Size(width, height); }
    void setWeights(const cv::Mat& weights);
    void setWeights(const vector<cv::Mat>& weights);
    void setOffset(const cv::Point& offset) { _offset = offset; }
    void setDCosts(const drwnDeformationCost& costs) { _dcost = costs; }
    const vector<cv::Mat>& getWeights() const { return _weights; }
    const cv::Point& getOffset() const { return _offset; }
    const drwnDeformationCost& getDCosts() const { return _dcost; }

    // inference (single and multi-channel variants)
    cv::Mat unaryCosts(const cv::Mat& features) const;
    cv::Mat unaryCosts(const vector<cv::Mat>& features) const;
    double pairwiseCost(const cv::Point& x, const cv::Point& c) const;

    // operators
    drwnPart& operator=(const drwnPart& part);

    // friends (overlap in pixel space)
    friend double overlap(const drwnPart& partA, const cv::Point& locationA,
        const drwnPart& partB, const cv::Point& locationB);
};

// drwnPartsModel -----------------------------------------------------------
//! Interface for implementing a part-based constellation model (i.e., pictorial
//! structures model) for object detection.
//!
//! Inference is done using distance transforms (on linear or quadratic
//! deformation costs) for efficiency.
//!
//! \warning The drwnPartsModel and associated classes are currently
//! experimental. The API and functionality may change in future versions.

class drwnPartsModel : public drwnStdObjIface {
 protected:
    cv::Size _baseSize;
    vector<drwnPart *> _parts;

 public:
    drwnPartsModel();
    drwnPartsModel(const drwnPartsModel& model);
    virtual ~drwnPartsModel();

    // i/o
    int numParts() const { return (int)_parts.size(); }
    const cv::Size& getBaseSize() const { return _baseSize; }
    void setBaseSize(const cv::Size& baseSize) { _baseSize = baseSize; }

    virtual void clear();
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);
    void swap(drwnPartsModel& model);

    virtual drwnPartsModel* clone() const = 0;

    // learning
    // ?

    // inference
    virtual double inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment) const;
    virtual double inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
        const cv::Point &centroidPrior) const;
    virtual double inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
        const vector<cv::Mat>& partPriors, const cv::Mat& centroidPrior) const;
    virtual double inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
        double& bestScale, double startScale, double endScale, int numLevels) const;
    //virtual double inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
    //    const vector<drwnPartsAssignment>& constraints) const;

    // negative max marginals (for each centroid location)
    cv::Mat energyLandscape(const cv::Mat& img) const;

    // populate list of objects
    void slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections) const;
    void slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections,
        int numLevelsPerOctave) const;

    // visualization
    cv::Mat showMAPPartLocations(const cv::Mat& img) const;
    cv::Mat showMAPPartLocations(const cv::Mat& img,
        const drwnPartsAssignment& assignment,
        double energy = DRWN_DBL_MAX, double scale = 1.0) const;
    cv::Mat showPartEnergyLandscape(const cv::Mat& img) const;

    // operators
    drwnPartsModel& operator=(const drwnPartsModel& model);
    drwnPart *operator[](unsigned indx) { return _parts[indx]; }
    const drwnPart *operator[](unsigned indx) const { return _parts[indx]; }

 protected:
    // computation of dense part matching costs
    virtual vector<cv::Mat> computeMatchingCosts(const cv::Mat& img) const = 0;

    // initial part locations
    static vector<pair<cv::Point, cv::Size> > initializePartLocations(int nParts,
        const cv::Size& imgSize);

    // colormap for visualization
    static cv::Scalar partColorMap(int v);
};

// drwnPartsModelMixture --------------------------------------------------
//! Mixture of parts models (T must have a parts model interface). Inference
//! returns the best scoring model and its parts locations.

template<typename T>
class drwnPartsModelMixture {
 protected:
    vector<T *> _models;

 public:
    drwnPartsModelMixture() { /* do nothing */ }
    ~drwnPartsModelMixture() {
        for (unsigned i = 0; i < _models.size(); i++) {
            delete _models[i];
        }
    }

    // i/o
    bool empty() const { return _models.empty(); }
    int size() const { return (int)_models.size(); }

    void clear() {
        for (int i = 0; i < (int)_models.size(); i++) {
            delete _models[i];
        }
        _models.clear();
    }

    void write(const char *filename) const {
        DRWN_ASSERT(filename != NULL);
        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnPartsModelMixture");
        for (unsigned i = 0; i < _models.size(); i++) {
            drwnXMLNode *child = drwnAddXMLChildNode(*node, "drwnPartsModel");
            _models[i]->save(*child);
        }
        ofstream ofs(filename);
        ofs << xml << endl;
        DRWN_ASSERT_MSG(!ofs.fail(), "could not write XML file " << filename);
        ofs.close();
    }

    void read(const char *filename) {
        DRWN_ASSERT(filename != NULL);
        clear();

        drwnXMLDoc xml;
        drwnParseXMLFile(xml, filename, "drwnPartsModelMixture");
        DRWN_ASSERT(!drwnIsXMLEmpty(xml));

        for (drwnXMLNode *node = xml.first_node("drwnPartsModel"); node != NULL;
             node = node->next_sibling("drwnPartsModel")) {
            DRWN_ASSERT(!drwnIsXMLEmpty(*node));

            T *m = new T();
            m->load(*node);
            add(m);
        }
    }

    // define models (add takes ownership, copy doesn't)
    void add(T *model) { _models.push_back(model); }
    void copy(const T *model) { _models.push_back(model->clone()); }

    // inference
    std::pair<int, double> inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment) const {
        DRWN_FCN_TIC;
        std::pair<int, double> bestModel(-1, DRWN_DBL_MAX);

        mapAssignment.clear();
        for (int i = 0; i < (int)_models.size(); i++) {
            drwnPartsAssignment assignment;
            double energy = _models[i]->inference(img, assignment);
            DRWN_LOG_DEBUG("...component " << i << " has energy " << energy);
            if (energy < bestModel.second) {
                bestModel = make_pair(i, energy);
                mapAssignment = assignment;
            }
        }

        DRWN_ASSERT(_models.empty() || (bestModel.first != -1));
        DRWN_FCN_TOC;
        return bestModel;
    }

    // inference with priors
    std::pair<int, double> inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
        const vector<vector<cv::Mat> >& partPriors, const cv::Mat& centroidPrior) const {
        DRWN_FCN_TIC;
        std::pair<int, double> bestModel(-1, DRWN_DBL_MAX);

        mapAssignment.clear();
        for (int i = 0; i < (int)_models.size(); i++) {
            drwnPartsAssignment assignment;
            double energy = _models[i]->inference(img, assignment, partPriors[i], centroidPrior);
            DRWN_LOG_DEBUG("...component " << i << " has energy " << energy);
            if (energy < bestModel.second) {
                bestModel = make_pair(i, energy);
                mapAssignment = assignment;
            }
        }

        DRWN_ASSERT(_models.empty() || (bestModel.first != -1));
        DRWN_FCN_TOC;
        return bestModel;
    }

    // inference at different scales
    std::pair<int, double> inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
        double& bestScale, double startScale, double endScale, int numLevels) const {

        DRWN_ASSERT((endScale <= startScale) && (numLevels > 0));
        const double scaleFactor = exp(log(endScale / startScale) / (double)numLevels);

        bestScale = 0.0;
        std::pair<int, double> bestModel(-1, DRWN_DBL_MAX);
        mapAssignment.clear();

        cv::Mat scaledImage(img.clone());
        drwnResizeInPlace(scaledImage, startScale * img.rows, startScale * img.cols);
        for (int i = 0; i < numLevels; i++) {
            drwnPartsAssignment assignment;
            std::pair<int, double> result = inference(scaledImage, assignment);
            if (result.second < bestModel.second) {
                bestModel = result;
                bestScale = (double)img.cols / (double)scaledImage.cols;
                mapAssignment = assignment;
            }

            if ((scaleFactor * scaledImage.rows < 1.0) ||
                (scaleFactor * scaledImage.cols < 1.0)) break;

            drwnResizeInPlace(scaledImage, scaleFactor * scaledImage.rows,
                scaleFactor * scaledImage.cols);
        }

        return bestModel;
    }

    // negative max marginals (for each centroid location)
    std::pair<cv::Mat, cv::Mat> energyLandscape(const cv::Mat& img) const {
        std::pair<cv::Mat, cv::Mat> energy;
        energy.first = cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(DRWN_FLT_MIN));
        energy.second = cv::Mat(img.rows, img.cols, CV_32SC1, cv::Scalar(-1));

        cv::Mat mask(img.rows, img.cols, CV_8UC1);
        for (int i = 0; i < (int)_models.size(); i++) {
            cv::Mat e = _models->energyLandscape(img);
            cv::compare(energy.first, e, mask, CV_CMP_GT);
            energy.first.setTo(e, mask);
            energy.second.setTo(cv::Scalar(i), mask);
        }

        return energy;
    }

    // populate list of objects (reference field set to model index)
    void slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections) const {
        for (int i = 0; i < (int)_models.size(); i++) {
            drwnObjectList d;
            _models[i]->slidingWindowDetections(img, d);
            for (drwnObjectList::iterator it = d.begin(); it != d.end(); ++it) {
                it->ref = i;
            }
            detections.insert(detections.end(), d.begin(), d.end());
        }
    }

    void slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections,
        int numLevelsPerOctave) const {
        for (int i = 0; i < (int)_models.size(); i++) {
            drwnObjectList d;
            _models[i]->slidingWindowDetections(img, d, numLevelsPerOctave);
            for (drwnObjectList::iterator it = d.begin(); it != d.end(); ++it) {
                it->ref = i;
            }
            detections.insert(detections.end(), d.begin(), d.end());
        }
    }


    // operators
    T* operator[](unsigned indx) { return _models[indx]; }
    const T* operator[](unsigned indx) const { return _models[indx]; }
};


// drwnTemplatePartsModel ---------------------------------------------------

class drwnTemplatePartsModel : public drwnPartsModel {
 public:
    drwnTemplatePartsModel();
    drwnTemplatePartsModel(const drwnTemplatePartsModel& model);
    virtual ~drwnTemplatePartsModel();

    // i/o
    const char *type() const { return "drwnTemplatePartsModel"; }
    drwnTemplatePartsModel *clone() const { return new drwnTemplatePartsModel(*this); }

    // learning
    void learnModel(int nParts, const vector<cv::Mat>& imgs);

 protected:
    // computation of dense part matching costs
    virtual vector<cv::Mat> computeMatchingCosts(const cv::Mat& img) const;
};

// drwnHOGPartsModel --------------------------------------------------------

class drwnHOGPartsModel : public drwnPartsModel {
 public:
    static int X_STEP_SIZE;
    static int Y_STEP_SIZE;

 public:
    drwnHOGPartsModel();
    drwnHOGPartsModel(const drwnHOGPartsModel& model);
    virtual ~drwnHOGPartsModel();

    // i/o
    const char *type() const { return "drwnHOGPartsModel"; }
    drwnHOGPartsModel *clone() const { return new drwnHOGPartsModel(*this); }

    // learning
    void learnModel(int nParts, const vector<cv::Mat>& imgs);
    void learnModel(const vector<cv::Size>& partSizes,
        const vector<pair<cv::Mat, drwnPartsAssignment> >& imgs);

 protected:
    // computation of dense part matching costs
    virtual vector<cv::Mat> computeMatchingCosts(const cv::Mat& img) const;
};

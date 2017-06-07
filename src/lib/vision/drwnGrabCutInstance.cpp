/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGrabCutInstance.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**		Kevin Guo <Kevin.Guo@nicta.com.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// opencv library headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"

#include "drwnGrabCutInstance.h"
#include "drwnOpenCVUtils.h"
#include "drwnVisionUtils.h"

using namespace std;
using namespace Eigen;

// define below if using Kolmogorov's maxflow-v3.01 (see external/installMaxFlow.pl)
#undef KOLMOGOROV_MAXFLOW
#ifdef KOLMOGOROV_MAXFLOW
#include "graph.h"
#endif

// drwnGrabCutInstance static members ---------------------------------------

#ifdef __APPLE__
const unsigned char drwnGrabCutInstance::MASK_FG;
const unsigned char drwnGrabCutInstance::MASK_BG;
const unsigned char drwnGrabCutInstance::MASK_C_FG;
const unsigned char drwnGrabCutInstance::MASK_C_BG;
const unsigned char drwnGrabCutInstance::MASK_C_BOTH;
const unsigned char drwnGrabCutInstance::MASK_C_NONE;
#endif

bool drwnGrabCutInstance::bVisualize = false;
int drwnGrabCutInstance::maxIterations = 10;

// drwnGrabCutInstance ------------------------------------------------------

drwnGrabCutInstance::drwnGrabCutInstance() :
    _numUnknown(0), _pairwise(NULL), _unaryWeight(1.0), _pottsWeight(0.0), _pairwiseWeight(0.0)
{
    // do nothing
}

drwnGrabCutInstance::drwnGrabCutInstance(const drwnGrabCutInstance& instance) :
    _img(instance._img.clone()), _trueMask(instance._trueMask.clone()), _mask(instance._mask.clone()),
    _numUnknown(instance._numUnknown), _unary(instance._unary.clone()), _pairwise(NULL),
    _unaryWeight(instance._unaryWeight), _pottsWeight(instance._pottsWeight), _pairwiseWeight(instance._pairwiseWeight)
{
    if (instance._pairwise != NULL) _pairwise = new drwnPixelNeighbourContrasts(*instance._pairwise);
}

drwnGrabCutInstance::~drwnGrabCutInstance()
{
    free();
}

cv::Mat drwnGrabCutInstance::knownForeground() const
{
    DRWN_ASSERT(_mask.data != NULL);
    cv::Mat mask(_mask.rows, _mask.cols, CV_8UC1);
    cv::compare(_mask, cv::Scalar(MASK_FG), mask, CV_CMP_EQ);
    return mask;
}

cv::Mat drwnGrabCutInstance::knownBackground() const
{
    DRWN_ASSERT(_mask.data != NULL);
    cv::Mat mask(_mask.rows, _mask.cols, CV_8UC1);
    cv::compare(_mask, cv::Scalar(MASK_BG), mask, CV_CMP_EQ);
    return mask;
}

cv::Mat drwnGrabCutInstance::unknownPixels() const
{
    DRWN_ASSERT(_mask.data != NULL);
    cv::Mat fgmask = knownForeground();
    cv::Mat bgmask = knownBackground();
    cv::Mat mask(_mask.rows, _mask.cols, CV_8UC1);
    cv::compare(fgmask, bgmask, mask, CV_CMP_EQ);
    return mask;
}

cv::Mat drwnGrabCutInstance::foregroundColourMask() const
{
    DRWN_ASSERT(_mask.data != NULL);
    cv::Mat mask = knownForeground();

    cv::Mat m(_mask.rows, _mask.cols, CV_8UC1);
    cv::compare(_mask, cv::Scalar(MASK_C_FG), m, CV_CMP_EQ);
    cv::bitwise_or(m, mask, mask);

    cv::compare(_mask, cv::Scalar(MASK_C_BOTH), m, CV_CMP_EQ);
    cv::bitwise_or(m, mask, mask);

    return mask;
}

cv::Mat drwnGrabCutInstance::backgroundColourMask() const
{
    DRWN_ASSERT(_mask.data != NULL);
    cv::Mat mask = knownBackground();

    cv::Mat m(_mask.rows, _mask.cols, CV_8UC1);
    cv::compare(_mask, cv::Scalar(MASK_C_BG), m, CV_CMP_EQ);
    cv::bitwise_or(m, mask, mask);

    cv::compare(_mask, cv::Scalar(MASK_C_BOTH), m, CV_CMP_EQ);
    cv::bitwise_or(m, mask, mask);

    return mask;
}

// initialization
void drwnGrabCutInstance::initialize(const cv::Mat& img, const cv::Rect& rect, const char *colorModelFile)
{
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(MASK_BG));
    mask(rect) = cv::Scalar(MASK_C_FG);
    initialize(img, mask, colorModelFile);
}

void drwnGrabCutInstance::initialize(const cv::Mat& img, const cv::Mat& inferMask, const char *colorModelFile)
{
    // create unknown true mask
    cv::Mat trueMask(img.rows, img.cols, CV_8UC1, cv::Scalar(MASK_C_FG));
    // initialize
    initialize(img, inferMask, trueMask, colorModelFile);
}

void drwnGrabCutInstance::initialize(const cv::Mat& img, const cv::Rect& rect,
    const cv::Mat& trueMask, const char *colorModelFile)
{
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(MASK_BG));
    mask(rect) = cv::Scalar(MASK_C_FG);
    initialize(img, mask, trueMask, colorModelFile);
}

void drwnGrabCutInstance::initialize(const cv::Mat& img, const cv::Mat& inferMask,
    const cv::Mat& trueMask, const char *colorModelFile)
{
    DRWN_ASSERT((img.rows == inferMask.rows) && (img.cols == inferMask.cols));
    DRWN_ASSERT((img.rows == trueMask.rows) && (img.cols == trueMask.cols));
    DRWN_ASSERT(img.type() == CV_8UC3);
    DRWN_ASSERT(inferMask.type() == CV_8UC1);
    DRWN_ASSERT(trueMask.type() == CV_8UC1);

    // delete previous instance data
    free();

    // clone image and masks
    _img = img.clone();
    _mask = inferMask.clone();
    _trueMask = trueMask.clone();

    _numUnknown = cv::countNonZero(unknownPixels());
    DRWN_LOG_VERBOSE(_numUnknown << " unknown pixels");

    // create unary potentials
    _unary = cv::Mat(img.rows, img.cols, CV_32FC1);

    // learn or load colour models
    if (colorModelFile == NULL) {
        learnColourModel(foregroundColourMask(), true);
        learnColourModel(backgroundColourMask(), false);
        updateUnaryPotentials();
    } else {
        DRWN_LOG_VERBOSE("loading colour models from " << colorModelFile);
        loadColourModels(colorModelFile);
    }

    // create pairwise potentials
    _pairwise = new drwnPixelNeighbourContrasts(_img);
}

void drwnGrabCutInstance::setBaseModelWeights(double u, double p, double c)
{
    DRWN_ASSERT_MSG(p >= 0.0, "potts term must be non-negative to remain submodular");
    DRWN_ASSERT_MSG(c >= 0.0, "contrast-sensitive pairwise term must be non-negative to remain submodular");

    const double scale = 1.0 / (double)std::max(numUnknown(), 1);
    _unaryWeight = scale * u;
    _pottsWeight = scale * p;
    _pairwiseWeight = scale * c;

    DRWN_LOG_VERBOSE("...setting model weights to " << _unaryWeight << ", " << _pottsWeight << ", " << _pairwiseWeight);
}

// energy
double drwnGrabCutInstance::unaryEnergy(const cv::Mat& seg) const
{
    double energy = 0.0;

    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            if (isUnknownPixel(x, y) && (seg.at<unsigned char>(y, x) == MASK_BG)) {
                energy += _unary.at<float>(y, x);
            }
        }
    }

    return energy;
}

double drwnGrabCutInstance::pottsEnergy(const cv::Mat& seg) const
{
    double energy = 0.0;

    // add horizontal potts terms
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 1; x < _img.cols; x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y, x - 1)) {
                energy += 1.0;
            }
        }
    }

    // add vertical potts terms
    for (int y = 1; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y - 1, x)) {
                energy += 1.0;
            }
        }
    }

    // add diagonal potts terms
    for (int y = 1; y < _img.rows; y++) {
        for (int x = 1; x < _img.cols; x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y - 1, x - 1)) {
                energy += M_SQRT1_2;
            }

            if (seg.at<unsigned char>(y, x - 1) != seg.at<unsigned char>(y - 1, x)) {
                energy += M_SQRT1_2;
            }
        }
    }

    return energy;
}

double drwnGrabCutInstance::pairwiseEnergy(const cv::Mat& seg) const
{
    double energy = 0.0;

    // add horizontal pairwise terms
    for (int y = 0; y < _pairwise->height(); y++) {
        for (int x = 1; x < _pairwise->width(); x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y, x - 1)) {
                energy += _pairwise->contrastW(x, y);
            }
        }
    }

    // add vertical pairwise terms
    for (int y = 1; y < _pairwise->height(); y++) {
        for (int x = 0; x < _pairwise->width(); x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y - 1, x)) {
                energy += _pairwise->contrastN(x, y);
            }
        }
    }

    // add diagonal pairwise terms
    for (int y = 1; y < _pairwise->height(); y++) {
        for (int x = 1; x < _pairwise->width(); x++) {
            if (seg.at<unsigned char>(y, x) != seg.at<unsigned char>(y - 1, x - 1)) {
                energy += _pairwise->contrastNW(x, y);
            }

            if (seg.at<unsigned char>(y, x - 1) != seg.at<unsigned char>(y - 1, x)) {
                energy += _pairwise->contrastSW(x, y - 1);
            }
        }
    }

    return energy;
}

double drwnGrabCutInstance::energy(const cv::Mat& seg) const
{
    return _unaryWeight * unaryEnergy(seg) +
        _pottsWeight * pottsEnergy(seg) +
        _pairwiseWeight * pairwiseEnergy(seg);
}

// percentage of pixels labeled as foreground
double drwnGrabCutInstance::foregroundRatio(const cv::Mat& seg) const
{
    DRWN_ASSERT((seg.rows == _mask.rows) && (seg.cols == _mask.cols));

    cv::Mat m = this->unknownPixels();
    cv::bitwise_and(seg, m, m);
    cv::compare(m, cv::Scalar(MASK_FG), m, CV_CMP_EQ);
    const int count = cv::countNonZero(m);

    return (double)count / (double)_numUnknown;
}

// loss
double drwnGrabCutInstance::loss(const cv::Mat& seg) const
{
    DRWN_ASSERT((seg.data != NULL) && (_trueMask.data != NULL));

    int numNotEqual = 0;
    int numCompared = 0;
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            if (isUnknownPixel(x, y) && !isUnknownPixel(x, y, _trueMask)) {
                if (seg.at<unsigned char>(y, x) != _trueMask.at<unsigned char>(y, x)) {
                    numNotEqual += 1;
                }
                numCompared += 1;
            }
        }
    }

    DRWN_ASSERT(numCompared > 0);
    return (double)numNotEqual / (double)numCompared;
}

cv::Mat drwnGrabCutInstance::inference()
{
    DRWN_FCN_TIC;

    // infer segmentation
    int nIteration = 0;
    cv::Mat lastSeg(_img.rows, _img.cols, CV_8UC1, cv::Scalar(0));

    cv::Mat mask(_img.rows, _img.cols, CV_8UC1);
    while (nIteration < maxIterations) {
        nIteration += 1;

        // run inference
        cv::Mat seg = graphCut(_unary);

        // visualize results
        if (bVisualize) {
            drwnShowDebuggingImage(seg, string("segmentation"), false);

            cv::Mat canvas = visualize(seg);
            drwnShowDebuggingImage(canvas, string("segmented_image"), false);
        }

        if (drwnCmpCount(seg, lastSeg, CV_CMP_NE) == 0) {
            break;
        }
        seg.copyTo(lastSeg);

        // update foreground and background models and unary potential
        if (nIteration < maxIterations) {
            cv::compare(lastSeg, cv::Scalar(MASK_FG), mask, CV_CMP_EQ);
            int count = cv::countNonZero(mask);
            if (count == 0) {
                DRWN_LOG_WARNING("segmentation is all background");
                break;
            }
            learnColourModel(mask, true);

            cv::compare(lastSeg, cv::Scalar(MASK_BG), mask, CV_CMP_EQ);
            count = cv::countNonZero(mask);
            if (count == 0) {
                DRWN_LOG_WARNING("segmentation is all foreground");
                break;
            }
            learnColourModel(mask, false);

            updateUnaryPotentials();
        }
    }

    if (nIteration < maxIterations) {
        DRWN_LOG_VERBOSE("segmentation for weight " << (_pairwiseWeight / _unaryWeight) << " converged in "
            << nIteration << " iterations");
    } else {
        DRWN_LOG_VERBOSE("segmentation for weight " << (_pairwiseWeight / _unaryWeight)
            << " did not converge in under " << maxIterations << " iterations");
    }

    DRWN_FCN_TOC;
    return lastSeg;
}

cv::Mat drwnGrabCutInstance::lossAugmentedInference()
{
    DRWN_FCN_TIC;

    // construct loss-augemented unary potentials
    cv::Mat lossAugmentedUnary(_unary.clone());

    // TODO: use "vectorized" version (e.g. cvCmpS)
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            // skip "known" pixels
            if (!isUnknownPixel(x, y)) continue;

            // check ground-truth
            switch (_trueMask.at<unsigned char>(y, x)) {
            case MASK_FG:
                lossAugmentedUnary.at<float>(y, x) -= 1.0f / (float)(_unaryWeight * _numUnknown);
                break;
            case MASK_BG:
                lossAugmentedUnary.at<float>(y, x) += 1.0f / (float)(_unaryWeight * _numUnknown);
                break;
            case MASK_C_FG:
            case MASK_C_BG:
            case MASK_C_BOTH:
            case MASK_C_NONE:
                break;
            default:
                DRWN_LOG_FATAL("invalid value in ground-truth mask");
            }
        }
    }

    // run inference
    cv::Mat seg = graphCut(lossAugmentedUnary);

    DRWN_FCN_TOC;
    return seg;
}

//! visualization
cv::Mat drwnGrabCutInstance::visualize(const cv::Mat& seg) const
{
    cv::Mat canvas(_img.clone());
    cv::Mat mask(canvas.rows, canvas.cols, CV_8UC1);
    cv::compare(seg, cv::Scalar(0x00), mask, CV_CMP_EQ);
    canvas.setTo(cv::Scalar(255, 0, 0), mask);
    return canvas;
}

void drwnGrabCutInstance::free()
{
    // delete instance data
    _img = cv::Mat();
    _mask = cv::Mat();
    _trueMask = cv::Mat();

    // delete cached data
    _unary = cv::Mat();
    if (_pairwise != NULL) { delete _pairwise; _pairwise = NULL; }
}

vector<double> drwnGrabCutInstance::pixelColour(int y, int x) const
{
    const unsigned char *p = _img.ptr<const unsigned char>(y) + 3 * x;
    vector<double> colour(3);
    colour[2] = (double)p[0] / 255.0;
    colour[1] = (double)p[1] / 255.0;
    colour[0] = (double)p[2] / 255.0;
    return colour;
}

cv::Mat drwnGrabCutInstance::graphCut(const cv::Mat& unary) const
{
    DRWN_ASSERT_MSG((unary.rows == _img.rows) && (unary.cols == _img.cols),
        "(" << unary.rows << " != " << _img.rows << ") || ("
        << unary.cols << " != " << _img.cols << ")");

    // number of random variables
    int nVariables = unary.rows * unary.cols;

    // construct s-t graph
    int hConstructGraph = drwnCodeProfiler::getHandle("graphCut.constructGraph");
    drwnCodeProfiler::tic(hConstructGraph);

#ifdef KOLMOGOROV_MAXFLOW
    typedef Graph<double, double, double> GraphType;
    GraphType *g = new GraphType(nVariables, 8 * nVariables);

    g->add_node(nVariables);
#else
    drwnBKMaxFlow g(nVariables);
    g.addNodes(nVariables);
#endif

    // add unary terms
    int varIndx = 0;
    for (int y = 0; y < unary.rows; y++) {
        for (int x = 0; x < unary.cols; x++, varIndx++) {
            switch (_mask.at<unsigned char>(y, x)) {
            case MASK_BG:
#ifdef KOLMOGOROV_MAXFLOW
                g->add_tweights(varIndx, 0.0, DRWN_FLT_MAX);
#else
                g.addTargetEdge(varIndx, DRWN_FLT_MAX);
#endif
                break;
            case MASK_FG:
#ifdef KOLMOGOROV_MAXFLOW
                g->add_tweights(varIndx, DRWN_FLT_MAX, 0.0);
#else
                g.addSourceEdge(varIndx, DRWN_FLT_MAX);
#endif
                break;
            default:
#ifdef KOLMOGOROV_MAXFLOW
                g->add_tweights(varIndx, _unaryWeight * (double)unary.at<float>(y, x), 0.0);
#else
                g.addSourceEdge(varIndx, _unaryWeight * (double)unary.at<float>(y, x));
#endif
            }
        }
    }

    // add horizontal pairwise terms
    for (int y = 0; y < _pairwise->height(); y++) {
        for (int x = 1; x < _pairwise->width(); x++) {
            int u = y * unary.cols + x;
            int v = y * unary.cols + x - 1;

            double w = _pottsWeight + _pairwiseWeight * _pairwise->contrastW(x, y);
            if (w > 0.0) {
#ifdef KOLMOGOROV_MAXFLOW
                g->add_edge(u, v, w, w);
#else
                g.addEdge(u, v, w, w);
#endif
            }
        }
    }

    // add vertical pairwise terms
    for (int y = 1; y < _pairwise->height(); y++) {
        for (int x = 0; x < _pairwise->width(); x++) {
            int u = y * unary.cols + x;
            int v = (y - 1) * unary.cols + x;

            double w = _pottsWeight + _pairwiseWeight * _pairwise->contrastN(x, y);
            if (w > 0.0) {
#ifdef KOLMOGOROV_MAXFLOW
                g->add_edge(u, v, w, w);
#else
                g.addEdge(u, v, w, w);
#endif
            }
        }
    }

    // add diagonal pairwise terms
    for (int y = 1; y < _pairwise->height(); y++) {
        for (int x = 1; x < _pairwise->width(); x++) {
            int u = y * unary.cols + x;
            int v = (y - 1) * unary.cols + x - 1;

            double w = _pottsWeight + _pairwiseWeight * _pairwise->contrastNW(x, y);
            if (w > 0.0) {
#ifdef KOLMOGOROV_MAXFLOW
                g->add_edge(u, v, w, w);
#else
                g.addEdge(u, v, w, w);
#endif
            }

            u = (y - 1) * unary.cols + x;
            v = y * unary.cols + x - 1;
            w = _pottsWeight + _pairwiseWeight * _pairwise->contrastSW(x, y - 1);
            if (w > 0.0) {
#ifdef KOLMOGOROV_MAXFLOW
                g->add_edge(u, v, w, w);
#else
                g.addEdge(u, v, w, w);
#endif
            }
        }
    }

    drwnCodeProfiler::toc(hConstructGraph);

    // run min-cut and decode solution
    int hMinCutIteration = drwnCodeProfiler::getHandle("graphCut.mincut");
    drwnCodeProfiler::tic(hMinCutIteration);
#ifdef KOLMOGOROV_MAXFLOW
    double e = g->maxflow();
#else
    double e = g.solve();
#endif
    DRWN_LOG_VERBOSE("...min-cut has value " << e);

    cv::Mat mapAssignment(unary.rows, unary.cols, CV_8UC1);
    varIndx = 0;
    for (int y = 0; y < unary.rows; y++) {
        unsigned char *p = mapAssignment.ptr<unsigned char>(y);
        for (int x = 0; x < unary.cols; x++, varIndx++) {
#ifdef KOLMOGOROV_MAXFLOW
            p[x] = (g->what_segment(varIndx) == GraphType::SOURCE) ? MASK_FG : MASK_BG;
#else
            p[x] = (!g.inSetT(varIndx) ? MASK_FG : MASK_BG);
#endif
        }
    }

    drwnCodeProfiler::toc(hMinCutIteration);
    return mapAssignment;
}

// drwnGrabCutInstanceGMM ---------------------------------------------------

size_t drwnGrabCutInstanceGMM::maxSamples = 5000;
int drwnGrabCutInstanceGMM::numMixtures = 5;

drwnGrabCutInstanceGMM::drwnGrabCutInstanceGMM() : drwnGrabCutInstance(),
     _fgColourModel(3, numMixtures), _bgColourModel(3, numMixtures)
{
    // do nothing
}

drwnGrabCutInstanceGMM::~drwnGrabCutInstanceGMM()
{
    // do nothing
}

// load colour models
void drwnGrabCutInstanceGMM::loadColourModels(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, filename, "drwnGrabCutInstanceGMM");

    drwnXMLNode *subnode = node->first_node("foreground");
    DRWN_ASSERT(subnode != NULL);
    _fgColourModel.load(*subnode);
    subnode = node->first_node("background");
    DRWN_ASSERT(subnode != NULL);
    _bgColourModel.load(*subnode);

    updateUnaryPotentials();
}

// save colour models
void drwnGrabCutInstanceGMM::saveColourModels(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnGrabCutInstanceGMM", NULL, false);
    drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);

    drwnXMLNode *child = drwnAddXMLChildNode(*node, "foreground", NULL, false);
    _fgColourModel.save(*child);

    child = drwnAddXMLChildNode(*node, "background", NULL, false);
    _bgColourModel.save(*child);

    ofstream ofs(filename);
    ofs << xml << endl;
    DRWN_ASSERT(!ofs.fail());
    ofs.close();
}

void drwnGrabCutInstanceGMM::learnColourModel(const cv::Mat& mask, bool bForeground)
{
    DRWN_ASSERT((mask.rows == _img.rows) && (mask.cols == _img.cols));
    if (maxSamples == 0) {
        DRWN_LOG_WARNING("skipping colour model learning (maxSamples is zero)");
        return;
    }

    DRWN_FCN_TIC;

    // extract colour samples for pixels in mask
    vector<vector<double> > data;
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            if (mask.at<unsigned char>(y, x) != 0x00) {
                data.push_back(pixelColour(y, x));
            }
        }
    }

    // subsample if too many
    data = drwn::subSample(data, maxSamples);
    DRWN_LOG_VERBOSE("learning " << numMixtures << "-component model using "
        << data.size() << " pixels...");

    // check variance of data
    drwnSuffStats stats(3, DRWN_PSS_FULL);
    stats.accumulate(data);
    VectorXd mu = stats.firstMoments() / stats.count();
    MatrixXd sigma = stats.secondMoments() / stats.count() - mu * mu.transpose();
    double det = sigma.determinant();
    if (det <= 0.0) {
        DRWN_LOG_WARNING("no colour variation in data; adding noise (|Sigma| = " << det << ")");
        for (unsigned i = 0; i < data.size(); i++) {
            data[i][0] += 0.01 * (drand48() - 0.5);
            data[i][1] += 0.01 * (drand48() - 0.5);
            data[i][2] += 0.01 * (drand48() - 0.5);
        }
    }

    // learn model
    DRWN_ASSERT(data.size() > 1);
    if (bForeground) {
        _fgColourModel.initialize(3, std::min(numMixtures, (int)data.size() - 1));
        _fgColourModel.train(data);
    } else {
        _bgColourModel.initialize(3, std::min(numMixtures, (int)data.size() - 1));
        _bgColourModel.train(data);
    }

    DRWN_FCN_TOC;
}

void drwnGrabCutInstanceGMM::updateUnaryPotentials()
{
    DRWN_ASSERT(_img.data != NULL);

    DRWN_FCN_TIC;
    DRWN_LOG_VERBOSE("updating unary potentials for " << toString(_img) << "...");
    if ((_unary.data == NULL) || (_unary.rows != _img.rows) || (_unary.cols != _img.cols)) {
        _unary = cv::Mat(_img.rows, _img.cols, CV_32FC1);
    }

    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            // skip "known" pixels
            if (!isUnknownPixel(x, y)) {
                _unary.at<float>(y, x) = 0.0f;
                continue;
            }

            // evaluate difference in log-likelihood
            vector<double> colour(pixelColour(y, x));

            double p_fg = _fgColourModel.evaluateSingle(colour);
            double p_bg = _bgColourModel.evaluateSingle(colour);
            DRWN_ASSERT(isfinite(p_fg) && isfinite(p_bg));
            _unary.at<float>(y, x) = (float)(p_fg - p_bg);
        }
    }

    DRWN_FCN_TOC;
}

// drwnGrabCutInstanceHistogram ---------------------------------------------

double drwnGrabCutInstanceHistogram::pseudoCounts = 1.0;
unsigned drwnGrabCutInstanceHistogram::channelBits = 3;

drwnGrabCutInstanceHistogram::drwnGrabCutInstanceHistogram() : drwnGrabCutInstance(),
    _fgColourModel(pseudoCounts, channelBits), _bgColourModel(pseudoCounts, channelBits)
{
    // do nothing
}

drwnGrabCutInstanceHistogram::~drwnGrabCutInstanceHistogram()
{
    // do nothing
}

// load colour models
void drwnGrabCutInstanceHistogram::loadColourModels(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, filename, "drwnGrabCutInstanceHistogram");

    drwnXMLNode *subnode = node->first_node("foreground");
    DRWN_ASSERT(subnode != NULL);
    _fgColourModel.load(*subnode);
    subnode = node->first_node("background");
    DRWN_ASSERT(subnode != NULL);
    _bgColourModel.load(*subnode);

    updateUnaryPotentials();
}

// save colour models
void drwnGrabCutInstanceHistogram::saveColourModels(const char *filename) const
{
    DRWN_ASSERT(filename != NULL);
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnGrabCutInstanceHistogram", NULL, false);
    drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);

    drwnXMLNode *child = drwnAddXMLChildNode(*node, "foreground", NULL, false);
    _fgColourModel.save(*child);
    child = drwnAddXMLChildNode(*node, "background", NULL, false);
    _bgColourModel.save(*child);

    ofstream ofs(filename);
    ofs << xml << endl;
    DRWN_ASSERT(!ofs.fail());
    ofs.close();
}

void drwnGrabCutInstanceHistogram::learnColourModel(const cv::Mat& mask, bool bForeground)
{
    DRWN_ASSERT((mask.rows == _img.rows) && (mask.cols == _img.cols));
    DRWN_FCN_TIC;

    drwnColourHistogram *model = (bForeground ? &_fgColourModel: &_bgColourModel);
    model->clear();

    // extract colour samples for pixels in mask
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            if (mask.at<unsigned char>(y, x) != 0x00) {
                const unsigned char *p = _img.ptr<const unsigned char>(y) + 3 * x;
                model->accumulate(p[2], p[1], p[0]);
            }

        }
    }

    DRWN_FCN_TOC;
}

void drwnGrabCutInstanceHistogram::updateUnaryPotentials()
{
    DRWN_ASSERT(_img.data != NULL);

    DRWN_FCN_TIC;
    DRWN_LOG_VERBOSE("updating unary potentials for " << toString(_img) << "...");
    if ((_unary.data == NULL) || (_unary.rows != _img.rows) || (_unary.cols != _img.cols)) {
        _unary = cv::Mat(_img.rows, _img.cols, CV_32FC1);
    }

    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            // skip "known" pixels
            if (!isUnknownPixel(x, y)) {
                _unary.at<float>(y, x) = 0.0f;
                continue;
            }

            // evaluate difference in log-likelihood
            const unsigned char *p = _img.ptr<const unsigned char>(y) + 3 * x;
            double p_fg = _fgColourModel.probability(p[2], p[1], p[0]);
            double p_bg = _bgColourModel.probability(p[2], p[1], p[0]);

            //assert probabilities are between 0 and 1
            DRWN_ASSERT((p_fg > 0.0) && (p_fg <= 1.0) && (p_bg > 0.0) && (p_bg <= 1.0));
            _unary.at<float>(y, x) = (float)(log(p_fg) - log(p_bg));
        }
    }

    DRWN_FCN_TOC;
}

// drwnGrabCutConfig --------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnGrabCut
//! \b visualize :: visualization (default: false)\n
//! \b maxIterations :: maximum segmentation iterations (default: 10)\n
//! \b maxSamples :: maximum samples for learning GMM colour models (default: 5000)\n
//! \b numMixtures :: number of mixture components in GMM colour models (default: 5)\n
//! \b pseudoCounts :: pseudo-counts for colour histogram models (default: 1.0)\n
//! \b channelBits :: number of bits per RGB colour channel in histogram colour models (default: 3)

class drwnGrabCutConfig : public drwnConfigurableModule {
public:
    drwnGrabCutConfig() : drwnConfigurableModule("drwnGrabCut") { }
    ~drwnGrabCutConfig() { }

    void usage(ostream &os) const {
        os << "      visualize       :: visualization\n";
        os << "      maxIterations   :: maximum segmentation iterations (default: "
           << drwnGrabCutInstance::maxIterations << ")\n";
        os << "      maxSamples      :: maximum samples for learning GMM colour models (default: "
           << drwnGrabCutInstanceGMM::maxSamples << ")\n";
        os << "      numMixtures     :: number of mixture components in GMM colour models (default: "
           << drwnGrabCutInstanceGMM::numMixtures << ")\n";
        os << "      pseudoCounts    :: pseudo-counts for colour histogram models (default: "
           << drwnGrabCutInstanceHistogram::pseudoCounts << ")\n";
        os << "      channelBits     ::  number of bits per RGB colour channel in histogram colour models (default: "
           << drwnGrabCutInstanceHistogram::channelBits << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "visualize")) {
            drwnGrabCutInstance::bVisualize = drwn::trueString(string(value));
        } else if (!strcmp(name, "maxIterations")) {
            drwnGrabCutInstance::maxIterations = std::max(0, atoi(value));
        } else if (!strcmp(name, "maxSamples")) {
            drwnGrabCutInstanceGMM::maxSamples = std::max(1, atoi(value));
        } else if (!strcmp(name, "numMixtures")) {
            drwnGrabCutInstanceGMM::numMixtures = std::max(1, atoi(value));
        } else if (!strcmp(name, "pseudoCounts")) {
            drwnGrabCutInstanceHistogram::pseudoCounts = std::max(0.0, atof(value));
        } else if (!strcmp(name, "channelBits")) {
            drwnGrabCutInstanceHistogram::channelBits = std::min(std::max(1, atoi(value)), 8);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnGrabCutConfig gGrabCutConfig;

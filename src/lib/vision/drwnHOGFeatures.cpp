/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnHOGFeatures.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <vector>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnHOGFeatures.h"
#include "drwnOpenCVUtils.h"

// drwnHOGFeatures static members -------------------------------------------

int drwnHOGFeatures::DEFAULT_CELL_SIZE = 8;
int drwnHOGFeatures::DEFAULT_BLOCK_SIZE = 2;
int drwnHOGFeatures::DEFAULT_BLOCK_STEP = 1;
int drwnHOGFeatures::DEFAULT_ORIENTATIONS = 9;
drwnHOGNormalization drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L2_NORM;
double drwnHOGFeatures::DEFAULT_CLIPPING_LB = 0.1;
double drwnHOGFeatures::DEFAULT_CLIPPING_UB = 0.5;
bool drwnHOGFeatures::DEFAULT_DIM_REDUCTION = false;

// drwnHOGFeatures ----------------------------------------------------------

drwnHOGFeatures::drwnHOGFeatures() :
    _cellSize(DEFAULT_CELL_SIZE), _blockSize(DEFAULT_BLOCK_SIZE),
    _blockStep(DEFAULT_BLOCK_STEP), _numOrientations(DEFAULT_ORIENTATIONS),
    _normalization(DEFAULT_NORMALIZATION),
    _clipping(DEFAULT_CLIPPING_LB, DEFAULT_CLIPPING_UB),
    _bDimReduction(DEFAULT_DIM_REDUCTION)
{
    // do nothing
}

drwnHOGFeatures::~drwnHOGFeatures()
{
    // do nothing
}

// gradient pre-processing (can be provided to computeFeatures)
pair<cv::Mat, cv::Mat> drwnHOGFeatures::gradientMagnitudeAndOrientation(const cv::Mat& img) const
{
    DRWN_ASSERT((img.data != NULL) && (img.channels() == 1));

    DRWN_LOG_DEBUG("Computing gradients on " << toString(img) << "...");
    cv::Mat Dx(img.rows, img.cols, CV_32FC1);
    cv::Sobel(img, Dx, CV_32F, 1, 0, 3);
    cv::Mat Dy(img.rows, img.cols, CV_32FC1);
    cv::Sobel(img, Dy, CV_32F, 0, 1, 3);

    // get canonical orientations for quantizing
    vector<float> u, v;
    computeCanonicalOrientations(u, v);

    // allocate memory for magintude (float) and orientation (int)
    cv::Mat gradMagnitude(img.rows, img.cols, CV_32FC1);
    cv::Mat gradOrientation(img.rows, img.cols, CV_32SC1);

    for (int y = 0; y < img.rows; y++) {
        const float *pDx = Dx.ptr<const float>(y);
        const float *pDy = Dy.ptr<const float>(y);
        float *pm = gradMagnitude.ptr<float>(y);
        int *po = gradOrientation.ptr<int>(y);
        for (int x = 0; x < img.cols; x++, pDx++, pDy++, pm++, po++) {
            // madnitude
            *pm = sqrt(((*pDx) * (*pDx)) + ((*pDy) * (*pDy)));

            // orientation
            *po = 0;
            float bestScore = fabs(u[0] * (*pDx) + v[0] * (*pDy));
            for (int i = 1; i < _numOrientations; i++) {
                float score = fabs(u[i] * (*pDx) + v[i] * (*pDy));
                if (score > bestScore) {
                    bestScore = score;
                    *po = i;
                }
            }
        }
    }

    return make_pair(gradMagnitude, gradOrientation);
}

// feature computation
void drwnHOGFeatures::computeFeatures(const cv::Mat& img, vector<cv::Mat>& features)
{
    // color convert
    DRWN_LOG_DEBUG("Color converting " << img.rows << "-by-" << img.cols << " image...");
    cv::Mat greyImg = drwnGreyImage(img);

    // compute and quantize gradients
    pair<cv::Mat, cv::Mat> magAndOri = gradientMagnitudeAndOrientation(greyImg);

    // compute actual features
    computeFeatures(magAndOri, features);
}

void drwnHOGFeatures::computeFeatures(const pair<cv::Mat, cv::Mat>& gradMagAndOri, vector<cv::Mat>& features)
{
    DRWN_FCN_TIC;
    const int NUM_FEATURES = numFeatures();

    // check input
    DRWN_ASSERT((gradMagAndOri.first.data != NULL) && (gradMagAndOri.second.data != NULL));
    DRWN_ASSERT((gradMagAndOri.first.rows == gradMagAndOri.second.rows) &&
        (gradMagAndOri.first.cols == gradMagAndOri.second.cols));
    DRWN_ASSERT(gradMagAndOri.first.type() == CV_32FC1);
    DRWN_ASSERT(gradMagAndOri.second.type() == CV_32SC1);

    if (features.empty()) {
        features.resize(NUM_FEATURES);
    }
    DRWN_ASSERT((int)features.size() == NUM_FEATURES);

    // compute cell histograms
    vector<cv::Mat> cellHistograms(_numOrientations);
    computeCellHistograms(gradMagAndOri, cellHistograms);

    // group into blocks and normalize
    vector<cv::Mat> *featuresPtr = NULL;
    vector<cv::Mat> fullFeatures;
    if (_bDimReduction) {
        fullFeatures.resize(_blockSize * _blockSize * _numOrientations);
        featuresPtr = &fullFeatures;
    } else {
        featuresPtr = &features;
    }
    computeBlockFeatures(cellHistograms, *featuresPtr);

    // normalize
    normalizeFeatureVectors(*featuresPtr);
    if ((_clipping.first > 0.0) || (_clipping.second < 1.0)) {
        clipFeatureVectors(*featuresPtr);
        normalizeFeatureVectors(*featuresPtr);
    }

    // reduce dimensionality
    if (_bDimReduction) {
        for (int i = 0; i < numFeatures(); i++) {
            if ((features[i].rows != fullFeatures[0].rows) || (features[i].cols != fullFeatures[0].cols)) {
                features[i] = cv::Mat::zeros(fullFeatures[0].rows, fullFeatures[0].cols, CV_32FC1);
            } else {
                features[i].setTo(cv::Scalar::all(0));
            }
        }

        // energy and orientation sums
        for (int i = 0; i < _numOrientations; i++) {
            for (int j = 0; j < _blockSize * _blockSize; j++) {
                cv::add(features[i], fullFeatures[j * _numOrientations + i], features[i]);
                cv::add(features[_numOrientations + j], fullFeatures[j * _numOrientations + i],
                    features[_numOrientations + j]);
            }
        }
    }

    DRWN_FCN_TOC;
}

// dense features are associated with the pixel at the center of the block
void drwnHOGFeatures::computeDenseFeatures(const cv::Mat& img, vector<cv::Mat >& features)
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(features.empty());

    // initialize features
    for (int i = 0; i < numFeatures(); i++) {
        features.push_back(cv::Mat::zeros(img.rows, img.cols, CV_32FC1));
    }

    // first pad image by blockSize/2 border
    cv::Mat greyImg = drwnGreyImage(img);
    cv::Mat paddedImg = drwnPadImage(greyImg, _blockSize * _cellSize);
    //drwnShowDebuggingImage(paddedImg, "paddedImage", true);

    // force single cell stepping
    const int oldBlockStep = _blockStep;
    _blockStep = 1;

    // compute gradients
    pair<cv::Mat, cv::Mat> magAndOri = gradientMagnitudeAndOrientation(paddedImg);

    // compute features with shifted origins
    vector<cv::Mat> responses(numFeatures());
    for (int y = 0; y < _cellSize; y++) {
        for (int x = 0; x < _cellSize; x++) {

            // shift gradients
            cv::Rect roi(x, y, greyImg.cols, greyImg.rows);
            cv::Mat shiftedMag = magAndOri.first(roi);
            cv::Mat shiftedOri = magAndOri.second(roi);

            // compute features
            computeFeatures(make_pair(shiftedMag, shiftedOri), responses);

            // copy into feature locations
            for (unsigned i = 0; i < responses.size(); i++) {
                for (int yy = 0; yy < responses[i].rows; yy++) {
                    if (yy * _cellSize + y >= greyImg.rows) break;
                    float *p = features[i].ptr<float>(yy * _cellSize + y) + x;
                    const float *q = responses[i].ptr<const float>(yy);
                    for (int xx = 0; xx < responses[i].cols; xx++) {
                        if (xx * _cellSize + x >= greyImg.cols) break;
                        p[xx * _cellSize] = q[xx];
                    }
                }
            }
        }
    }

    // restore state
    _blockStep = oldBlockStep;

    DRWN_FCN_TOC;
}

// visualization
cv::Mat drwnHOGFeatures::visualizeCells(const cv::Mat& img, int scale)
{
    if (scale < 1) scale = 1;

    // TODO: refactor to share computation
    const int oldSize = _blockSize;
    const int oldStep = _blockStep;

    _blockSize = 1;
    _blockStep = 1;

    vector<cv::Mat> features;
    computeFeatures(img, features);
    DRWN_ASSERT(features.size() == (unsigned)_numOrientations);

    vector<float> u, v;
    computeCanonicalOrientations(u, v);

    const int numCellsX = features[0].cols;
    const int numCellsY = features[0].rows;

    cv::Mat canvas = cv::Mat::zeros(scale * numCellsY * _cellSize, scale * numCellsX * _cellSize, CV_8UC3);

    for (int y = 0; y < numCellsY; y++) {
        int cy = scale * (y * _cellSize + _cellSize/2);
        for (int x = 0; x < numCellsX; x++) {
            int cx = scale * (x * _cellSize + _cellSize/2);

            multimap<float, int> sortedOrientations;
            for (int o = 0; o < _numOrientations; o++) {
                sortedOrientations.insert(make_pair(features[o].at<float>(y, x), o));
            }

            // draw orientations in sorted order
            for (multimap<float, int>::const_iterator it = sortedOrientations.begin();
                 it != sortedOrientations.end(); it++) {
                int strength = (int)(255.0 * it->first);
                int o = it->second;

                // rotate gradient orientations by 90 degrees
                cv::line(canvas, cv::Point((int)(cx - 0.9 * scale * v[o] * _cellSize/2),
                        (int)(cy + 0.9 * scale * u[o] * _cellSize/2)),
                    cv::Point((int)(cx + 0.9 * scale * v[o] * _cellSize/2),
                        (int)(cy - 0.9 * scale * u[o] * _cellSize/2)),
                    cv::Scalar::all(strength), std::min(2, scale));
            }
        }
    }

    _blockSize = oldSize;
    _blockStep = oldStep;
    return canvas;
}

// compute the representative orientation for each bin
void drwnHOGFeatures::computeCanonicalOrientations(vector<float>& x, vector<float>& y) const
{
    x.resize(_numOrientations);
    y.resize(_numOrientations);

    double theta = 0.0;
    for (int i = 0; i < _numOrientations; i++) {
        x[i] = cos(theta);
        y[i] = sin(theta);
        theta += M_PI / (_numOrientations + 1.0);
    }
    DRWN_ASSERT(theta <= M_PI);
}

// compute histograms of oriented gradients for each cell
void drwnHOGFeatures::computeCellHistograms(const pair<cv::Mat, cv::Mat>& gradMagAndOri,
    vector<cv::Mat>& cellHistograms) const
{
    // check input
    const cv::Mat gradMagnitude(gradMagAndOri.first);
    const cv::Mat gradOrientation(gradMagAndOri.second);
    DRWN_ASSERT((gradMagnitude.data != NULL) && (gradOrientation.data != NULL));
    DRWN_ASSERT((gradMagnitude.rows == gradOrientation.rows) &&
        (gradMagnitude.cols == gradOrientation.cols));

    const int width = gradMagnitude.cols;
    const int height = gradMagnitude.rows;

    // compute cell histograms
    const int numCellsX = (width + _cellSize - 1) / _cellSize;
    const int numCellsY = (height + _cellSize - 1) / _cellSize;
    DRWN_LOG_DEBUG("Computing histograms for each " << numCellsX << "-by-" << numCellsY << " cell...");

    DRWN_ASSERT(cellHistograms.size() == (unsigned)_numOrientations);
    for (int i = 0; i < _numOrientations; i++) {
        if ((cellHistograms[i].rows != numCellsY) || (cellHistograms[i].cols != numCellsX)) {
            cellHistograms[i] = cv::Mat::zeros(numCellsY, numCellsX, CV_32FC1);
        } else {
            cellHistograms[i].setTo(cv::Scalar::all(0));
        }
    }

    // vote for cell using interpolation
    for (int y = 0; y < height; y++) {
        const float *pm = gradMagnitude.ptr<const float>(y);
        const int *po = gradOrientation.ptr<int>(y);
        for (int x = 0; x < width; x++) {

            float cellXCoord = ((float)x + 0.5) / (float)_cellSize - 0.5;
            float cellYCoord = ((float)y + 0.5) / (float)_cellSize - 0.5;

            int cellXIndx = (int)cellXCoord; // integer cell index
            int cellYIndx = (int)cellYCoord;

            cellXCoord -= (float)cellXIndx; // fractional cell index
            cellYCoord -= (float)cellYIndx;

            float *pc = cellHistograms[po[x]].ptr<float>(cellYIndx) + cellXIndx;

            if ((cellXIndx >= 0) && (cellYIndx >= 0)) {
                pc[0] += (1.0f - cellXCoord) * (1.0f - cellYCoord) * pm[x];
            }

            if ((cellXIndx >= 0) && (cellYIndx + 1 < numCellsY)) {
                pc[numCellsX] += (1.0f - cellXCoord) * cellYCoord * pm[x];
            }

            if ((cellXIndx + 1 < numCellsX) && (cellYIndx + 1 < numCellsY)) {
                pc[numCellsX + 1] += cellXCoord * cellYCoord * pm[x];
            }

            if ((cellXIndx + 1 < numCellsX) && (cellYIndx >= 0)) {
                pc[1] += cellXCoord * (1.0f - cellYCoord) * pm[x];
            }
        }
    }
}

void drwnHOGFeatures::computeBlockFeatures(const vector<cv::Mat>& cellHistograms, vector<cv::Mat>& features) const
{
    const int NUM_FEATURES = _blockSize * _blockSize * _numOrientations;
    DRWN_ASSERT(features.size() == (unsigned)NUM_FEATURES);
    DRWN_ASSERT(cellHistograms.size() == (unsigned)_numOrientations);

    // group cell histograms into blocks
    const int numCellsX = cellHistograms.front().cols;
    const int numCellsY = cellHistograms.front().rows;
    const int numBlocksX = (numCellsX - _blockSize + 1) / _blockStep;
    const int numBlocksY = (numCellsY - _blockSize + 1) / _blockStep;

    DRWN_LOG_DEBUG("Computing and normalizing feature vectors for " <<
        numBlocksX << "-by-" << numBlocksY << " blocks...");

    // allocate memory for blocks
    for (int i = 0; i < NUM_FEATURES; i++) {
        if ((features[i].rows != numBlocksY) || (features[i].cols != numBlocksX)) {
            features[i] = cv::Mat(numBlocksY, numBlocksX, CV_32FC1);
        }
    }

    // compute features
    for (int y = 0; y < numBlocksY; y++) {
        for (int x = 0; x < numBlocksX; x++) {

            int featureIndx = 0;
            for (int cy = _blockStep * y; cy < _blockStep * y + _blockSize; cy++) {
                for (int cx = _blockStep * x; cx < _blockStep * x + _blockSize; cx++) {
                    for (int o = 0; o < _numOrientations; o++) {
                        features[featureIndx].at<float>(y, x) = cellHistograms[o].at<float>(cy, cx);
                        featureIndx += 1;
                    }
                }
            }
        }
    }
}

// normalization
void drwnHOGFeatures::normalizeFeatureVectors(vector<cv::Mat>& features) const
{
    if (features.empty()) return;

    const int height = features[0].rows;
    const int width = features[0].cols;

    // compute normalization constant
    cv::Mat Z(height, width, CV_32FC1, cv::Scalar(DRWN_EPSILON));
    cv::Mat tmp(height, width, CV_32FC1);

    for (unsigned i = 0; i < features.size(); i++) {
        switch (_normalization) {
        case DRWN_HOG_L2_NORM:
            cv::multiply(features[i], features[i], tmp);
            break;
        case DRWN_HOG_L1_NORM:
        case DRWN_HOG_L1_SQRT:
            tmp = cv::abs(features[i]);
            break;
        default:
            DRWN_LOG_FATAL("unknown normalization method");
        }

        Z += tmp;
    }

    if (_normalization == DRWN_HOG_L2_NORM) {
        cv::sqrt(Z, Z);
    }

    // perform normalization
    for (unsigned i = 0; i < features.size(); i++) {
        cv::divide(features[i], Z, features[i]);
    }

    if (_normalization == DRWN_HOG_L1_SQRT) {
        for (unsigned i = 0; i < features.size(); i++) {
            cv::sqrt(features[i], features[i]);
        }
    }
}

// clip feature values to [C.lb, C.ub] and rescale to [0, 1]
void drwnHOGFeatures::clipFeatureVectors(std::vector<cv::Mat>& features) const
{
    DRWN_ASSERT(_clipping.first < _clipping.second);
    for (unsigned i = 0; i < features.size(); i++) {
        cv::max(features[i], cv::Scalar(_clipping.first), features[i]);
        cv::min(features[i], cv::Scalar(_clipping.second), features[i]);
        features[i] = features[i] / (_clipping.second - _clipping.first) -
            _clipping.first / (_clipping.second - _clipping.first);
    }
}

// drwnHOGFeaturesConfig ----------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnHOGFeatures
//! \b cellSize        :: cell size in pixels (default: 8)\n
//! \b blockSize       :: block size in cells (default: 2)\n
//! \b blockStep       :: block step in cells (default: 1)\n
//! \b numOrientations :: number of orientations (default: 9)\n
//! \b normMethod      :: normalization method (L2_NORM (default), L1_NORM, L1_SQRT)\n
//! \b normClippingLB  :: lower clipping after normalization (default: 0.1)\n
//! \b normClippingUB  :: upper clipping after normalization (default: 0.5)\n
//! \b dimReduction    :: analytic dimensionality reduction (default: false)

class drwnHOGFeaturesConfig : public drwnConfigurableModule {
public:
    drwnHOGFeaturesConfig() : drwnConfigurableModule("drwnHOGFeatures") { }
    ~drwnHOGFeaturesConfig() { }

    void usage(ostream &os) const {
        os << "      cellSize        :: cell size in pixels (default: "
           << drwnHOGFeatures::DEFAULT_CELL_SIZE << ")\n";
        os << "      blockSize       :: block size in cells (default: "
           << drwnHOGFeatures::DEFAULT_BLOCK_SIZE << ")\n";
        os << "      blockStep       :: block step in cells (default: "
           << drwnHOGFeatures::DEFAULT_BLOCK_STEP << ")\n";
        os << "      numOrientations :: number of orientations (default: "
           << drwnHOGFeatures::DEFAULT_ORIENTATIONS << ")\n";
        os << "      normMethod      :: normalization method (L2_NORM (default), L1_NORM, L1_SQRT)\n";
        os << "      normClippingLB  :: lower clipping after normalization (default: "
           << drwnHOGFeatures::DEFAULT_CLIPPING_LB << ")\n";
        os << "      normClippingUB  :: upper clipping after normalization (default: "
           << drwnHOGFeatures::DEFAULT_CLIPPING_UB << ")\n";
        os << "      dimReduction    :: analytic dimensionality reduction (default: "
           << (drwnHOGFeatures::DEFAULT_DIM_REDUCTION ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "cellSize")) {
            drwnHOGFeatures::DEFAULT_CELL_SIZE = std::max(1, atoi(value));
        } else if (!strcmp(name, "blockSize")) {
            drwnHOGFeatures::DEFAULT_BLOCK_SIZE = std::max(1, atoi(value));
        } else if (!strcmp(name, "blockStep")) {
            drwnHOGFeatures::DEFAULT_BLOCK_STEP = std::max(1, atoi(value));
        } else if (!strcmp(name, "numOrientations")) {
            drwnHOGFeatures::DEFAULT_ORIENTATIONS = std::max(1, atoi(value));
        } else if (!strcmp(name, "normMethod")) {
            if (!strcasecmp(value, "L2_NORM")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L2_NORM;
            } else if (!strcasecmp(value, "L1_NORM")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L1_NORM;
            } else if (!strcasecmp(value, "L1_SQRT")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L1_SQRT;
            } else {
                DRWN_LOG_FATAL("unrecognized configuration value " << value
                    << " for option " << name << " in " << this->name());
            }
        } else if (!strcmp(name, "normClippingLB")) {
            drwnHOGFeatures::DEFAULT_CLIPPING_LB = std::max(0.0, atof(value));
        } else if (!strcmp(name, "normClippingUB")) {
            drwnHOGFeatures::DEFAULT_CLIPPING_UB = std::min(1.0, atof(value));
        } else if (!strcmp(name, "dimReduction")) {
            drwnHOGFeatures::DEFAULT_DIM_REDUCTION = trueString(string(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnHOGFeaturesConfig gHOGFeaturesConfig;

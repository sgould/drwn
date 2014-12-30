/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnOpenCVUtils.h"
#include "drwnOpenCVNodes.h"

using namespace std;
using namespace Eigen;

// drwnOpenCVImageSourceNode -------------------------------------------------------

drwnOpenCVImageSourceNode::drwnOpenCVImageSourceNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _extension(".png")
{
    _nVersion = 100;
    _desc = "Imports images as matrices";

    // define ports
    _outputPorts.push_back(new drwnOutputPort(this, "imageOut", "N-by-3 matrix of image data"));
    _outputPorts.push_back(new drwnOutputPort(this, "sizeOut", "size of image"));

    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
}

drwnOpenCVImageSourceNode::drwnOpenCVImageSourceNode(const drwnOpenCVImageSourceNode& node) :
    drwnNode(node), _directory(node._directory), _extension(node._extension)
{
    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
}

drwnOpenCVImageSourceNode::~drwnOpenCVImageSourceNode()
{
    // do nothing
}

// processing
void drwnOpenCVImageSourceNode::evaluateForwards()
{
    // clear output tables and then update forwards
    clearOutput();
    updateForwards();
}

void drwnOpenCVImageSourceNode::updateForwards()
{
    drwnDataTable *dataOut = _outputPorts[0]->getTable();
    drwnDataTable *sizeOut = _outputPorts[1]->getTable();
    DRWN_ASSERT((dataOut != NULL) && (sizeOut != NULL));

    // list all matching files in the directory
    vector<string> keys = drwnDirectoryListing(_directory.c_str(),
        _extension.c_str(), false, false);
    if (keys.empty()) {
        DRWN_LOG_WARNING("no files found in \"" << _directory << "\" for node \"" << _name << "\"");
        return;
    }

    // import data
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        if (dataOut->hasKey(*it)) continue;

        // load data into record
        string filename = _directory + DRWN_DIRSEP + *it + _extension;
        IplImage *img = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_COLOR);
        if (img == NULL) {
            DRWN_LOG_ERROR("failed to load image \"" << filename << "\"");
            continue;
        }

        drwnDataRecord *dataRecOut = dataOut->lockRecord(*it);
        drwnDataRecord *sizeRecOut = sizeOut->lockRecord(*it);

        dataRecOut->structure().resize(2);
        dataRecOut->structure() << img->height, img->width;
        dataRecOut->data() = MatrixXd::Zero(img->width * img->height, 3);

        sizeRecOut->data() = MatrixXd::Zero(1, 2);
        sizeRecOut->data()[0] = (double)img->width;
        sizeRecOut->data()[1] = (double)img->height;

        int indx = 0;
        const unsigned char *p = (const unsigned char *)img->imageData;
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++, indx++) {
                dataRecOut->data()(indx, 0) = (double)p[3 * x + 2];
                dataRecOut->data()(indx, 1) = (double)p[3 * x + 1];
                dataRecOut->data()(indx, 2) = (double)p[3 * x + 0];
            }
            p += img->widthStep;
        }

        // release memory and records
        cvReleaseImage(&img);
        dataOut->unlockRecord(*it);
        sizeOut->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

// drwnOpenCVImageSinkNode ---------------------------------------------------------

drwnOpenCVImageSinkNode::drwnOpenCVImageSinkNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _extension(".png")
{
    _nVersion = 100;
    _desc = "Exports matrices as images";

    // define ports
    _inputPorts.push_back(new drwnInputPort(this, "imageIn", "N-by-3 or N-by-1 matrix of image data"));

    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
}

drwnOpenCVImageSinkNode::drwnOpenCVImageSinkNode(const drwnOpenCVImageSinkNode& node) :
    drwnNode(node), _directory(node._directory), _extension(node._extension)
{
    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
}

drwnOpenCVImageSinkNode::~drwnOpenCVImageSinkNode()
{
    // do nothing
}

// processing
void drwnOpenCVImageSinkNode::evaluateForwards()
{
    drwnDataTable *dataIn = _inputPorts[0]->getTable();
    if (dataIn == NULL) {
        DRWN_LOG_ERROR("input port not connected for node \"" << _name << "\"");
        return;
    }

    // export data
    vector<string> keys = dataIn->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;

        string filename = _directory + DRWN_DIRSEP + *it + _extension;

        // export image
        drwnDataRecord *dataRecIn = dataIn->lockRecord(*it);
        exportImage(filename, dataRecIn);

        // release records
        dataIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnOpenCVImageSinkNode::updateForwards()
{
    drwnDataTable *dataIn = _inputPorts[0]->getTable();
    if (dataIn == NULL) {
        DRWN_LOG_ERROR("input port not connected for node \"" << _name << "\"");
        return;
    }

    // export data
    vector<string> keys = dataIn->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;

        string filename = _directory + DRWN_DIRSEP + *it + _extension;

        // don't overwrite existing output records
        if (drwnFileExists(filename.c_str())) continue;

        // export image
        drwnDataRecord *dataRecIn = dataIn->lockRecord(*it);
        exportImage(filename, dataRecIn);

        // release records
        dataIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnOpenCVImageSinkNode::exportImage(const string &filename,
    const drwnDataRecord *dataRecIn) const
{
    // convert record to image
    DRWN_ASSERT(dataRecIn->structure().size() == 2);

    int height = dataRecIn->structure()[0];
    int width = dataRecIn->structure()[1];

    DRWN_ASSERT((dataRecIn->numFeatures() == 1) || (dataRecIn->numFeatures() == 3));
    DRWN_ASSERT(dataRecIn->numObservations() == width * height);

    bool bColour = (dataRecIn->numFeatures() == 3);
    IplImage *img = cvCreateImage(cvSize(width, height),
        IPL_DEPTH_8U, bColour ? 3 : 1);

    int indx = 0;
    unsigned char *p = (unsigned char *)img->imageData;
    if (bColour) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++, indx++) {
                p[3 * x + 2] = (unsigned char)dataRecIn->data()(indx, 0);
                p[3 * x + 1] = (unsigned char)dataRecIn->data()(indx, 1);
                p[3 * x + 0] = (unsigned char)dataRecIn->data()(indx, 2);
            }
            p += img->widthStep;
        }
    } else {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++, indx++) {
                p[x] = (unsigned char)dataRecIn->data()[indx];
            }
            p += img->widthStep;
        }
    }

    // save image
    cvSaveImage(filename.c_str(), img);
    cvReleaseImage(&img);
}

// drwnOpenCVResizeNode ------------------------------------------------------

vector<string> drwnOpenCVResizeNode::_interpolationMethods;

drwnOpenCVResizeNode::drwnOpenCVResizeNode(const char *name, drwnGraph *owner) :
    drwnMultiIONode(name, owner), _defaultWidth(320), _defaultHeight(240), _interpolation(1)
{
    _nVersion = 100;
    _desc = "Resizes a multi-channel image";

    // define interpolation methods if not already done
    if (_interpolationMethods.empty()) {
        _interpolationMethods.push_back(string("nearest-neighbour"));
        _interpolationMethods.push_back(string("bilinear"));
        _interpolationMethods.push_back(string("area"));
        _interpolationMethods.push_back(string("bicubic"));
    }

    // declare propertys
    declareProperty("width", new drwnIntegerProperty(&_defaultWidth));
    declareProperty("height", new drwnIntegerProperty(&_defaultHeight));
    declareProperty("interpolation", new drwnSelectionProperty(&_interpolation, &_interpolationMethods));

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "imageIn", "N-by-D image data matrix"));
    _inputPorts.push_back(new drwnInputPort(this, "newSizeIn", "1-by-2 output image size"));
    _outputPorts.push_back(new drwnOutputPort(this, "imageOut", "M-by-D image data matrix"));
}

drwnOpenCVResizeNode::drwnOpenCVResizeNode(const drwnOpenCVResizeNode& node) :
    drwnMultiIONode(node), _defaultWidth(node._defaultWidth),
    _defaultHeight(node._defaultHeight), _interpolation(node._interpolation)
{
    // declare propertys
    declareProperty("width", new drwnIntegerProperty(&_defaultWidth));
    declareProperty("height", new drwnIntegerProperty(&_defaultHeight));
    declareProperty("interpolation", new drwnSelectionProperty(&_interpolation, &_interpolationMethods));
}

drwnOpenCVResizeNode::~drwnOpenCVResizeNode()
{
    // do nothing
}

// processing
bool drwnOpenCVResizeNode::forwardFunction(const string& key,
    const vector<const drwnDataRecord *>& src, const vector<drwnDataRecord *>& dst)
{
    DRWN_ASSERT((src.size() == 2) && (dst.size() == 1));

    // check for input
    if ((src[0] == NULL) || !src[0]->hasData())
        return false;

    DRWN_ASSERT(src[0]->structure().size() == 2);
    DRWN_ASSERT(src[0]->numObservations() == src[0]->structure()[0] * src[0]->structure()[1]);

    // create opencv matrices
    CvMat *dataIn = cvCreateMat(src[0]->structure()[0], src[0]->structure()[1], CV_32FC1);

    CvMat *dataOut = NULL;
    if ((src[1] == NULL) || !src[1]->hasData()) {
        DRWN_LOG_DEBUG("using default size " << _defaultHeight << "-by-" << _defaultWidth
            << " for record " << key);
        dataOut = cvCreateMat(_defaultHeight, _defaultWidth, CV_32FC1);
    } else {
        DRWN_ASSERT((src[1]->numObservations() == 1) && (src[1]->numFeatures() == 2));
        DRWN_LOG_DEBUG("using input size " << src[1]->data()[1] << "-by-" << src[1]->data()[0]
            << " for record " << key);
        dataOut = cvCreateMat((int)src[1]->data()[1], (int)src[1]->data()[0], CV_32FC1);
    }

    dst[0]->structure().resize(2);
    dst[0]->structure() << dataOut->rows, dataOut->cols;
    dst[0]->data() = MatrixXd::Zero(dataOut->rows * dataOut->cols, src[0]->numFeatures());

    // resize each channel
    int resizeMethod = CV_INTER_LINEAR;
    switch (_interpolation) {
    case 0: resizeMethod = CV_INTER_NN; break;
    case 1: resizeMethod = CV_INTER_LINEAR; break;
    case 2: resizeMethod = CV_INTER_AREA; break;
    case 3: resizeMethod = CV_INTER_CUBIC; break;
    default:
        DRWN_LOG_ERROR("unrecognized interpolation method " << _interpolation);
    }

    DRWN_LOG_DEBUG("processing " << src[0]->numFeatures() << "-channel image of size "
        << dataIn->cols << "-by-" << dataIn->rows);

    for (int i = 0; i < src[0]->numFeatures(); i++) {
        Eigen::Map<MatrixXf>((float *)dataIn->data.ptr,
            dataIn->rows * dataIn->cols, 1) = src[0]->data().col(i).cast<float>();
        cvResize(dataIn, dataOut, resizeMethod);
        dst[0]->data().col(i) = Eigen::Map<MatrixXf>((float *)dataOut->data.ptr,
            dataOut->rows * dataOut->cols, 1).cast<double>();
    }

    // free opencv data structures
    cvReleaseMat(&dataIn);
    cvReleaseMat(&dataOut);

    return true;
}

bool drwnOpenCVResizeNode::backwardGradient(const string& key,
    const vector<drwnDataRecord *>& src, const vector<const drwnDataRecord *>& dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// drwnOpenCVFilterBankNode --------------------------------------------------------

drwnOpenCVFilterBankNode::drwnOpenCVFilterBankNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner), _kappa(1.0)
{
    _nVersion = 100;
    _desc = "17-dimensional (texton) filter bank";

    // define ports
    _inputPorts[0]->setDescription("N-by-3 or N-by-1 matrix of image data");
    _outputPorts[0]->setDescription("N-by-17 filter response matrix");

    // declare propertys
    declareProperty("kappa", new drwnDoubleProperty(&_kappa));
}

drwnOpenCVFilterBankNode::drwnOpenCVFilterBankNode(const drwnOpenCVFilterBankNode& node) :
    drwnSimpleNode(node), _kappa(node._kappa)
{
    // declare propertys
    declareProperty("kappa", new drwnDoubleProperty(&_kappa));
}

drwnOpenCVFilterBankNode::~drwnOpenCVFilterBankNode()
{
    // do nothing
}

// processing
bool drwnOpenCVFilterBankNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    DRWN_ASSERT(dst != NULL);
    const int NUM_FILTERS = 17;

    // check input
    if ((src == NULL) || !src->hasData())
        return false;

    IplImage *img = record2Image(src);
    if (img == NULL) {
        return false;
    }

    // create output
    dst->structure() = src->structure();
    dst->data() = MatrixXd::Zero(img->height * img->width, NUM_FILTERS);

    // colour convert image
    IplImage *imgCIELab = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 3);
    if (img->nChannels == 1) {
        IplImage *tmp = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
        cvCvtColor(img, tmp, CV_GRAY2BGR);
        cvConvertScale(tmp, imgCIELab, 1.0 / 255.0, 0.0);
        cvReleaseImage(&tmp);
    } else {
        DRWN_ASSERT(img->nChannels == 3);
        cvConvertScale(img, imgCIELab, 1.0 / 255.0, 0.0);
    }
    cvCvtColor(imgCIELab, imgCIELab, CV_BGR2Lab);

    IplImage *greyImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
    cvSetImageCOI(imgCIELab, 1);
    cvCopyImage(imgCIELab, greyImg);
    cvSetImageCOI(imgCIELab, 0);

    // temporary response matrix
    CvMat *response = cvCreateMat(img->height, img->width, CV_32FC1);

    int k = 0;

    // gaussian filter on all color channels
    DRWN_LOG_DEBUG("Generating gaussian filter responses...");
    IplImage *imgChannel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
    for (double sigma = 1.0; sigma <= 4.0; sigma *= 2.0) {
        for (int c = 1; c <= 3; c++) {
            cvSetImageCOI(imgCIELab, c);
            cvCopyImage(imgCIELab, imgChannel);
            cvSmooth(imgChannel, response, CV_GAUSSIAN, 2 * (int)(_kappa * sigma) + 1);
            dst->data().col(k) = Eigen::Map<MatrixXf>((float *)response->data.ptr,
                response->rows * response->cols, 1).cast<double>();
            k += 1;
        }
        cvSetImageCOI(imgCIELab, 0);
    }
    cvReleaseImage(&imgChannel);

    // derivatives of gaussians on just greyscale image
    DRWN_LOG_DEBUG("Generating derivative of gaussian filter responses...");
    for (double sigma = 2.0; sigma <= 4.0; sigma *= 2.0) {
        // x-direction
        cvSobel(greyImg, response, 1, 0, 1);
        cvSmooth(response, response, CV_GAUSSIAN,
            2 * (int)(_kappa * sigma) + 1, 2 * (int)(3.0 * _kappa * sigma) + 1);
        dst->data().col(k) = Eigen::Map<MatrixXf>((float *)response->data.ptr,
            response->rows * response->cols, 1).cast<double>();
        k += 1;

        // y-direction
        cvSobel(greyImg, response, 0, 1, 1);
        cvSmooth(response, response, CV_GAUSSIAN,
            2 * (int)(3.0 * _kappa * sigma) + 1, 2 * (int)(_kappa * sigma) + 1);
        dst->data().col(k) = Eigen::Map<MatrixXf>((float *)response->data.ptr,
            response->rows * response->cols, 1).cast<double>();
        k += 1;
    }

    // laplacian of gaussian on just greyscale image
    DRWN_LOG_DEBUG("Generating laplacian of gaussian filter responses...");
    IplImage *tmpImg = cvCreateImage(cvGetSize(greyImg), IPL_DEPTH_32F, 1);
    for (double sigma = 1.0; sigma <= 8.0; sigma *= 2.0) {
        cvSmooth(greyImg, tmpImg, CV_GAUSSIAN, 2 * (int)(_kappa * sigma) + 1);
        cvLaplace(tmpImg, response);
        dst->data().col(k) = Eigen::Map<MatrixXf>((float *)response->data.ptr,
            response->rows * response->cols, 1).cast<double>();
        k += 1;
    }

    // free memory
    cvReleaseMat(&response);
    cvReleaseImage(&tmpImg);
    cvReleaseImage(&greyImg);
    cvReleaseImage(&imgCIELab);
    cvReleaseImage(&img);

    DRWN_ASSERT_MSG(k == NUM_FILTERS, k << " != " << NUM_FILTERS);
    return true;
}

bool drwnOpenCVFilterBankNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// drwnOpenCVIntegralImageNode -----------------------------------------------------

drwnOpenCVIntegralImageNode::drwnOpenCVIntegralImageNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner)
{
    _nVersion = 100;
    _desc = "produces integral images (for faster computation)";

    // define ports
    _inputPorts[0]->setDescription("H-by-W-by-K matrix of features");
    _outputPorts[0]->setDescription("(H+1)-by-(W+1)-by-K matrix of integral features");
}

drwnOpenCVIntegralImageNode::drwnOpenCVIntegralImageNode(const drwnOpenCVIntegralImageNode& node) :
    drwnSimpleNode(node)
{
    // do nothing
}

drwnOpenCVIntegralImageNode::~drwnOpenCVIntegralImageNode()
{
    // do nothing
}

// processing
bool drwnOpenCVIntegralImageNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    DRWN_ASSERT(dst != NULL);

    // check input
    if ((src == NULL) || !src->hasData())
        return false;

    // create output
    if (src->hasStructure()) {
        dst->structure() = src->structure().array() + 1;
        dst->data() = MatrixXd::Zero(dst->structure()[0] * dst->structure()[1],
            src->numFeatures());

        // TODO
        DRWN_ASSERT(false);
    } else {
        dst->data() = MatrixXd::Zero(src->numObservations() + 1,
            src->numFeatures());
        for (int i = 0; i < src->numObservations(); i++) {
            dst->data().row(i + 1) = dst->data().row(i) + src->data().row(i);
        }
    }

    return true;
}

bool drwnOpenCVIntegralImageNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// drwnOpenCVNeighborhoodFeaturesNode ----------------------------------------

drwnOpenCVNeighborhoodFeaturesNode::drwnOpenCVNeighborhoodFeaturesNode(const char *name, drwnGraph *owner) :
    drwnSimpleNode(name, owner), _cellSize(3), _bIncludeRow(true), _bIncludeCol(true)
{
    _nVersion = 100;
    _desc = "computes summary features in neighborhood around each pixel";

    // define ports
    _inputPorts[0]->setDescription("H-by-W-by-K matrix of features");
    _outputPorts[0]->setDescription("H-by-W-by-19K matrix of summary features");

    // declare propertys
    declareProperty("cellSize", new drwnRangeProperty(&_cellSize, 1, DRWN_INT_MAX));
    declareProperty("includeRow", new drwnBooleanProperty(&_bIncludeRow));
    declareProperty("includeCol", new drwnBooleanProperty(&_bIncludeCol));
}

drwnOpenCVNeighborhoodFeaturesNode::drwnOpenCVNeighborhoodFeaturesNode(const drwnOpenCVNeighborhoodFeaturesNode& node) :
    drwnSimpleNode(node), _cellSize(node._cellSize), _bIncludeRow(node._bIncludeRow), _bIncludeCol(node._bIncludeCol)
{
    // declare propertys
    declareProperty("cellSize", new drwnRangeProperty(&_cellSize, 1, DRWN_INT_MAX));
    declareProperty("includeRow", new drwnBooleanProperty(&_bIncludeRow));
    declareProperty("includeCol", new drwnBooleanProperty(&_bIncludeCol));
}

drwnOpenCVNeighborhoodFeaturesNode::~drwnOpenCVNeighborhoodFeaturesNode()
{
    // do nothing
}

// processing
bool drwnOpenCVNeighborhoodFeaturesNode::forwardFunction(const string& key,
    const drwnDataRecord *src, drwnDataRecord *dst)
{
    DRWN_ASSERT(dst != NULL);

    // check input
    if ((src == NULL) || !src->hasData())
        return false;

    // create output
    if (src->hasStructure()) {
        dst->structure() = src->structure();
        int height = src->structure()[0];
        int width = src->structure()[1];
        int nFeatures = 19 * src->numFeatures();
        if (_bIncludeRow) nFeatures += 1;
        if (_bIncludeCol) nFeatures += 1;
        dst->data() = MatrixXd::Zero(height * width, nFeatures);

        // copy base features
        dst->data().block(0, 0, height * width, src->numFeatures()) = src->data();

        // extract features as matrices
        vector<CvMat *> rawFeatures = record2CvMats(src);
        CvMat *rawSum = cvCreateMat(height + 1, width + 1, CV_64FC1);
        CvMat *rawSqSum = cvCreateMat(height + 1, width + 1, CV_64FC1);

        // iterate over features
        for (int i = 0; i < (int)rawFeatures.size(); i++) {
            // compute integral image for feature
            cvIntegral(rawFeatures[i], rawSum, rawSqSum);

            // compute mean and standard deviation around each pixel
            int rowIndx = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++, rowIndx++) {

                    int colIndx = 18 * i + src->numFeatures();
                    for (int dy = (int)(-1.5 * _cellSize); dy < _cellSize; dy += _cellSize) {
                        for (int dx = (int)(-1.5 * _cellSize); dx < _cellSize; dx += _cellSize) {
                            CvRect roi = cvRect(x + dx, y + dy, _cellSize, _cellSize);
                            drwnClipRect(roi, height, width);

                            if ((roi.width == 0) || (roi.height == 0)) {
                                dst->data()(rowIndx, colIndx) = src->data()(rowIndx, i);
                                dst->data()(rowIndx, colIndx + 1) = 0.0;
                            } else {

                                // compute mean and variance
                                double mu = CV_MAT_ELEM(*rawSum, double, roi.y, roi.x) +
                                    CV_MAT_ELEM(*rawSum, double, roi.y + roi.height, roi.x + roi.width) -
                                    CV_MAT_ELEM(*rawSum, double, roi.y + roi.height, roi.x) -
                                    CV_MAT_ELEM(*rawSum, double, roi.y, roi.x + roi.width);
                                mu /= (double)(roi.width * roi.height);

                                double sigma = CV_MAT_ELEM(*rawSqSum, double, roi.y, roi.x) +
                                    CV_MAT_ELEM(*rawSqSum, double, roi.y + roi.height, roi.x + roi.width) -
                                    CV_MAT_ELEM(*rawSqSum, double, roi.y + roi.height, roi.x) -
                                    CV_MAT_ELEM(*rawSqSum, double, roi.y, roi.x + roi.width);
                                sigma = std::max(0.0, sigma / (double)(roi.width * roi.height) - mu * mu);

                                dst->data()(rowIndx, colIndx) = mu;
                                dst->data()(rowIndx, colIndx + 1) = sqrt(sigma);
                            }

                            // move to next output column
                            colIndx += 2;
                        }
                    }
                }
            }
        }

        // add row and column features
        int rowIndx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++, rowIndx++) {
                int colIndx = 19 * src->numFeatures();
                if (_bIncludeRow) {
                    dst->data()(rowIndx, colIndx) = 2.0 * (double)y / (double)height - 1.0;
                    colIndx += 1;
                }
                if (_bIncludeCol) {
                    dst->data()(rowIndx, colIndx) = 2.0 * (double)x / (double)width - 1.0;
                }
            }
        }

        // free memory
        cvReleaseMat(&rawSqSum);
        cvReleaseMat(&rawSum);
        releaseOpenCVMatrices(rawFeatures);

    } else {
        DRWN_LOG_ERROR(getName() << " requires structured data");
        return false;
    }

    return true;
}

bool drwnOpenCVNeighborhoodFeaturesNode::backwardGradient(const string& key,
    drwnDataRecord *src, const drwnDataRecord *dst)
{
    DRWN_NOT_IMPLEMENTED_YET;
    return false;
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVImageSourceNode);
DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVImageSinkNode);
DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVResizeNode);
DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVFilterBankNode);
DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVIntegralImageNode);
DRWN_AUTOREGISTERNODE("OpenCV", drwnOpenCVNeighborhoodFeaturesNode);

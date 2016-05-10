/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNNGraph.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

#include "drwnNNGraph.h"

using namespace std;
using namespace Eigen;

// drwnNNGraphNode -----------------------------------------------------------

void drwnNNGraphNode::clear()
{
    features = VectorXf();
    label = -1;
    edges.clear();
    spatialNeighbours.clear();
}

bool drwnNNGraphNode::insert(const drwnNNGraphEdge& e)
{
    // don't update if new match does not improve the score
    if (edges.empty() || (e.weight >= edges.back().weight))
        return false;

    // search for existing match to the same target image id
    for (drwnNNGraphEdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
        if (it->targetNode.imgIndx == e.targetNode.imgIndx) {
            if (it->weight > e.weight) {
                edges.erase(it);
                drwnNNGraphEdgeList tmp(1, e);
                edges.merge(tmp, drwnNNGraphSortByScore());
                return true;
            } else {
                return false;
            }
        }
    }

    // otherwise remove the previous worst match and add this one
    edges.pop_back();
    drwnNNGraphEdgeList tmp(1, e);
    edges.merge(tmp, drwnNNGraphSortByScore());

    return true;
}

size_t drwnNNGraphNode::numBytesOnDisk() const
{
    size_t n = 0;
    n += sizeof(uint16_t) + features.size() * sizeof(float);
    n += sizeof(int32_t);
    n += sizeof(uint16_t) + edges.size() * (2 * sizeof(uint16_t) + sizeof(float));
    n += sizeof(uint16_t) + spatialNeighbours.size() * sizeof(uint16_t);
    return n;
}

bool drwnNNGraphNode::read(istream& is)
{
    clear();

    // read features
    uint16_t n;
    is.read((char *)&n, sizeof(uint16_t));
    features = VectorXf(n);
    is.read((char *)&features[0], n * sizeof(float));

    // read labels
    is.read((char *)&label, sizeof(int32_t));

    // read edges
    is.read((char *)&n, sizeof(uint16_t));
    while (n-- > 0) {
        drwnNNGraphEdge e;
        is.read((char *)&e.targetNode.imgIndx, sizeof(uint16_t));
        is.read((char *)&e.targetNode.segId, sizeof(uint16_t));
        is.read((char *)&e.weight, sizeof(float));
        edges.push_back(e);
    }

    // read spatial neighbours
    is.read((char *)&n, sizeof(uint16_t));
    if (n > 0) {
        vector<uint16_t> data(n);
        is.read((char *)&data[0], n * sizeof(uint16_t));
        spatialNeighbours.insert(data.begin(), data.end());
    }

    return true;
}

bool drwnNNGraphNode::write(ostream& os) const
{
    // write features
    uint16_t n = features.size();
    os.write((char *)&n, sizeof(uint16_t));
    os.write((char *)&features[0], n * sizeof(float));

    // write labels
    os.write((char *)&label, sizeof(int32_t));

    // write edges
    n = edges.size();
    os.write((char *)&n, sizeof(uint16_t));
    for (drwnNNGraphEdgeList::const_iterator it = edges.begin(); it != edges.end(); ++it) {
        os.write((char *)&it->targetNode.imgIndx, sizeof(uint16_t));
        os.write((char *)&it->targetNode.segId, sizeof(uint16_t));
        os.write((char *)&it->weight, sizeof(float));
    }

    // write spatial neighbours
    n = spatialNeighbours.size();
    os.write((char *)&n, sizeof(uint16_t));
    if (n > 0) {
        vector<uint16_t> data(spatialNeighbours.begin(), spatialNeighbours.end());
        os.write((char *)&data[0], n * sizeof(uint16_t));
    }

    return true;
}

// drwnNNGraphImageData ------------------------------------------------------

string drwnNNGraphImageData::imgDir("data/images");
string drwnNNGraphImageData::imgExt(".jpg");
string drwnNNGraphImageData::lblDir("data/labels");
string drwnNNGraphImageData::lblExt(".txt");
string drwnNNGraphImageData::segDir("data/regions");
string drwnNNGraphImageData::segExt(".bin");

drwnNNGraphImageData::drwnNNGraphImageData(const string &name) : _name(name)
{
    // load image
    string filename = imgDir + DRWN_DIRSEP + _name + imgExt;
    _img = cv::imread(filename, cv::IMREAD_COLOR);
    DRWN_ASSERT_MSG(_img.data != NULL, filename);

    // load labels
    _labels = MatrixXi::Constant(_img.rows, _img.cols, -1);
    filename = lblDir + DRWN_DIRSEP + _name + lblExt;
    if (drwnFileExists(filename.c_str())) {
        //drwnLoadPixelLabels(_labels, filename.c_str());
        _labels = gMultiSegRegionDefs.loadLabelFile(filename.c_str());
	DRWN_ASSERT((_labels.rows() == _img.rows) && (_labels.cols() == _img.cols));
    }

    // load segments
    filename = segDir + DRWN_DIRSEP + _name + segExt;
    DRWN_ASSERT_MSG(drwnFileExists(filename.c_str()), filename << " does not exists");

    ifstream ifs(filename.c_str(), ios::binary);
    _segments.read(ifs);
    ifs.close();
    cacheSegmentData();
}

drwnNNGraphImageData::drwnNNGraphImageData(const cv::Mat& img,
    const drwnSuperpixelContainer& segments) : _name(""), _img(img), _segments(segments)
{
    DRWN_ASSERT((_img.data != NULL) && (_img.channels() == 3) && (_img.depth() == CV_8U));
    _labels = MatrixXi::Constant(_img.rows, _img.cols, -1);
    cacheSegmentData();
}

void drwnNNGraphImageData::setLabels(const MatrixXi& labels)
{
    DRWN_ASSERT(((size_t)labels.rows() == height()) && ((size_t)labels.cols() == width()));
    _labels = labels;
}

vector<VectorXd> drwnNNGraphImageData::getSegmentLabelMarginals(int numLabels) const
{
    numLabels = std::max((int)_labels.maxCoeff() + 1, numLabels);
    if (numLabels < 1) {
        return vector<VectorXd>(numSegments());
    }

    // compute marginals
    vector<VectorXd> marginals(numSegments(), VectorXd::Zero(numLabels));

    for (int c = 0; c < _segments.channels(); c++) {
        for (int y = 0; y < _segments.height(); y++) {
            for (int x = 0; x < _segments.width(); x++) {
                const int segId = _segments[c].at<int>(y, x);
                if (segId < 0) continue;
                const int lblId = _labels(y, x);
                if (lblId < 0) {
                    marginals[segId] += VectorXd::Constant(numLabels, 1.0 / (double)numLabels);
                } else {
                    marginals[segId](lblId) += 1.0;
                }
            }
        }
    }

    // normalize
    for (unsigned segId = 0; segId < marginals.size(); segId++) {
        const double Z = marginals[segId].sum();
        if (Z != 0.0) {
            marginals[segId] /= Z;
        }
    }

    return marginals;
}

void drwnNNGraphImageData::cacheSegmentData()
{
    DRWN_ASSERT((_segments.width() == _img.cols) && (_segments.height() == _img.rows));

    // cache segment colours
    _colours.resize(_segments.size());
    vector<Vector3d> accRGB(_segments.size(), Vector3d::Zero());
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            for (int c = 0; c < _segments.channels(); c++) {
                const int segId = _segments[c].at<int>(y, x);
                if (segId < 0) continue;

                accRGB[segId][0] += (double)_img.at<unsigned char>(y, _img.channels() * x);
                accRGB[segId][1] += (double)_img.at<unsigned char>(y, _img.channels() * x + 1);
                accRGB[segId][2] += (double)_img.at<unsigned char>(y, _img.channels() * x + 2);
            }
        }
    }

    for (int segId = 0; segId < _segments.size(); segId++) {
        unsigned red = (unsigned)(accRGB[segId][2] / _segments.pixels(segId));
        unsigned green = (unsigned)(accRGB[segId][1] / _segments.pixels(segId));
        unsigned blue = (unsigned)(accRGB[segId][0] / _segments.pixels(segId));
        _colours[segId] = (blue << 16) | (green << 8) | red;
    }

    // cache segment centroids
    _centroids.resize(_segments.size());
    vector<Vector2i> accXY(_segments.size(), Vector2i::Zero());
    for (int y = 0; y < _img.rows; y++) {
        for (int x = 0; x < _img.cols; x++) {
            for (int c = 0; c < _segments.channels(); c++) {
                const int segId = _segments[c].at<int>(y, x);
                if (segId < 0) continue;

                accXY[segId].x() += x;
                accXY[segId].y() += y;
            }
        }
    }

    for (int segId = 0; segId < _segments.size(); segId++) {
        _centroids[segId].x = accXY[segId].x() / _segments.pixels(segId);
        _centroids[segId].y = accXY[segId].y() / _segments.pixels(segId);
    }
}

// drwnNNGraphImage ----------------------------------------------------------

drwnNNGraphImage::drwnNNGraphImage(const string& name, unsigned n) :
    drwnPersistentRecord(), bSourceMatchable(true), bTargetMatchable(true), eqvClass(-1), _name("")
{
    initialize(name, n);
}

drwnNNGraphImage::drwnNNGraphImage(const drwnNNGraphImageData& image) :
    drwnPersistentRecord(), bSourceMatchable(true), bTargetMatchable(true), eqvClass(-1), _name("")
{
    initialize(image);
}

void drwnNNGraphImage::clearNodes()
{
    _nodes.clear();
}

void drwnNNGraphImage::clearEdges()
{
    for (vector<drwnNNGraphNode>::iterator it = _nodes.begin(); it != _nodes.end(); ++it) {
        it->edges.clear();
    }
}

void drwnNNGraphImage::initialize(const string& name, unsigned n)
{
    _name = name;
    _nodes.clear();
    _nodes.resize(n);
}

void drwnNNGraphImage::initialize(const drwnNNGraphImageData& image)
{
    // basic meta-data
    _name = image.name();
    _nodes.clear();
    _nodes.resize(image.segments().size());

    // node neighbourhoods
    cacheNodeNeighbourhoods(image);

    // node features and labels
    cacheNodeFeatures(image);
    cacheNodeLabels(image);
}

void drwnNNGraphImage::transformNodeFeatures(const drwnFeatureTransform& xform)
{
    vector<double> x, y;
    for (size_t i = 0; i < _nodes.size(); i++) {
        x.resize(_nodes[i].features.rows());
        Eigen::Map<VectorXd>(&x[0], x.size()) = _nodes[i].features.cast<double>();
        xform.transform(x, y);
        _nodes[i].features = Eigen::Map<VectorXd>(&y[0], y.size()).cast<float>();
    }
}

void drwnNNGraphImage::appendNodeFeatures(const drwnNNGraphImageData& image, const cv::Mat& features)
{
    DRWN_ASSERT(((int)image.width() == features.cols) && ((int)image.height() == features.rows));
    DRWN_ASSERT(image.numSegments() == this->numNodes());

    // convert to 32-bit floating point (if not already)
    if (features.depth() != CV_32F) {
        cv::Mat tmp(features.rows, features.cols, CV_8U);
        features.convertTo(tmp, CV_32F, 1.0, 0.0);
        return appendNodeFeatures(image, tmp);
    }

    // compute mean pixel feature over each superpixel
    vector<float> phi(image.numSegments(), 0.0f);
    for (unsigned y = 0; y < image.height(); y++) {
        for (unsigned x = 0; x < image.width(); x++) {
            const float p = features.at<float>(y, x);
            for (int c = 0; c < image.segments().channels(); c++) {
                const int segId = image.segments()[c].at<int>(y, x);
                if (segId < 0) continue;

                phi[segId] += p;
            }
        }
    }

    for (unsigned segId = 0; segId < phi.size(); segId++) {
        DRWN_ASSERT(isfinite(phi[segId]));
        VectorXf newFeatures(_nodes[segId].features.rows() + 1);
        newFeatures.head(_nodes[segId].features.rows()) = _nodes[segId].features;
        newFeatures[_nodes[segId].features.rows()] = phi[segId] / (float)image.segments().pixels(segId);
        _nodes[segId].features = newFeatures;
    }
}

void drwnNNGraphImage::appendNodeFeatures(const drwnNNGraphImageData& image, const vector<cv::Mat>& features)
{
    for (unsigned i = 0; i < features.size(); i++) {
        appendNodeFeatures(image, features[i]);
    }
}

size_t drwnNNGraphImage::numBytesOnDisk() const
{
    // meta data
    size_t n = 0;
    n += sizeof(uint16_t) + _name.size() * sizeof(unsigned char);
    n += sizeof(int32_t);

    // nodes
    n += sizeof(uint16_t);
    for (size_t i = 0; i < _nodes.size(); i++) {
        n += _nodes[i].numBytesOnDisk();
    }

    return n;
}

bool drwnNNGraphImage::read(istream& is)
{
    // clear existing data
    clearNodes();

    // read meta data
    uint16_t n;
    is.read((char *)&n, sizeof(uint16_t));
    char *name = new char[n + 1];
    is.read(name, n * sizeof(char));
    name[n] = '\0';
    _name = string(name);
    delete[] name;

    is.read((char *)&eqvClass, sizeof(int32_t));

    // read nodes
    is.read((char *)&n, sizeof(uint16_t));
    _nodes.resize(n);
    for (size_t i = 0; i < _nodes.size(); i++) {
        _nodes[i].read(is);
    }

    return true;
}

bool drwnNNGraphImage::write(ostream& os) const
{
    // write meta data
    uint16_t n = _name.size();
    os.write((char *)&n, sizeof(uint16_t));
    os.write(_name.c_str(), n * sizeof(char));
    os.write((char *)&eqvClass, sizeof(int32_t));

    // write nodes
    n = _nodes.size();
    os.write((char *)&n, sizeof(uint16_t));
    for (size_t i = 0; i < _nodes.size(); i++) {
        _nodes[i].write(os);
    }

    return true;
}

drwnNNGraphImage drwnNNGraphImage::clone(bool bWithFeatures) const
{
    drwnNNGraphImage img(_name, _nodes.size());
    for (unsigned i = 0; i < _nodes.size(); i++) {
        if (bWithFeatures) {
            img[i].features = _nodes[i].features;
        }
        img[i].label = _nodes[i].label;
        img[i].edges = _nodes[i].edges;
        img[i].spatialNeighbours = _nodes[i].spatialNeighbours;
    }
    return img;
}

void drwnNNGraphImage::cacheNodeNeighbourhoods(const drwnNNGraphImageData& image)
{
    // add adjacent superpixels in the same layer
    for (int c = 0; c < image.segments().channels(); c++) {
        for (unsigned y = 0; y < image.height(); y++) {
            const int *p = image.segments()[c].ptr<const int>(y);
            const int *q = image.segments()[c].ptr<const int>(std::max((int)y - 1, 0));
            for (unsigned x = 0; x < image.width(); x++) {
                if ((p[x] != q[x]) && (p[x] >= 0) && (q[x] >= 0)) {
                    _nodes[p[x]].spatialNeighbours.insert(q[x]);
                    _nodes[q[x]].spatialNeighbours.insert(p[x]);
                }
                if ((x > 0) && (p[x] != p[x - 1]) && (p[x] >= 0) && (p[x - 1] >= 0)) {
                    _nodes[p[x]].spatialNeighbours.insert(p[x - 1]);
                    _nodes[p[x - 1]].spatialNeighbours.insert(p[x]);
                }
            }
        }
    }

    // add overlapping superpixels (from different layers)
    for (int c = 0; c < image.segments().channels() - 1; c++) {
        for (unsigned y = 0; y < image.height(); y++) {
            const int *p = image.segments()[c].ptr<const int>(y);
            const int *q = image.segments()[c + 1].ptr<const int>(y);
            for (unsigned x = 0; x < image.width(); x++) {
                if ((p[x] >= 0) && (q[x] >= 0)) {
                    _nodes[p[x]].spatialNeighbours.insert(q[x]);
                    _nodes[q[x]].spatialNeighbours.insert(p[x]);
                }
            }
        }
    }
}

void drwnNNGraphImage::cacheNodeFeatures(const drwnNNGraphImageData& image)
{
    DRWN_ASSERT((int)_nodes.size() == image.segments().size());

    // compute pixel features
    drwnTextonFilterBank textonBank(1.0);
    vector<cv::Mat> responses;
    textonBank.filter(image.image(), responses);

    // add vertical offset and horizontal deviation
    responses.push_back(responses[0].clone());
    responses.push_back(responses[0].clone());
    for (unsigned y = 0; y < image.height(); y++) {
        float *p = responses[responses.size() - 2].ptr<float>(y);
        float *q = responses[responses.size() - 1].ptr<float>(y);
        for (unsigned x = 0; x < image.width(); x++) {
            p[x] = (float)y / image.height();
            q[x] = 2.0f * fabs((float)x / image.width() - 0.5f);
        }
    }

    // add dense HOG features
    drwnHOGFeatures::DEFAULT_BLOCK_SIZE = 2;
    drwnHOGFeatures::DEFAULT_DIM_REDUCTION = true;
    drwnHOGFeatures hogFeatureBank;
    vector<cv::Mat> hogResponses;
    hogFeatureBank.computeDenseFeatures(image.image(), hogResponses);
    responses.insert(responses.end(), hogResponses.begin(), hogResponses.end());

    // add LBP features
    drwnLBPFilterBank lbpFilterBank(false);
    vector<cv::Mat> lbpResponses;
    lbpFilterBank.filter(image.image(), lbpResponses);
    responses.insert(responses.end(), lbpResponses.begin(), lbpResponses.end());

    // entropy feature
    vector<vector<int> > qcolour(3 * _nodes.size(), vector<int>(256, 0));
    for (unsigned y = 0; y < image.height(); y++) {
        const unsigned char *p = image.image().ptr<const unsigned char>(y);
        for (unsigned x = 0; x < image.width(); x++) {
            for (int c = 0; c < image.segments().channels(); c++) {
                const int segId = image.segments()[c].at<int>(y, x);
                if (segId < 0) continue;
                qcolour[3 * segId][p[3 * x]] += 1;
                qcolour[3 * segId + 1][p[3 * x + 1]] += 1;
                qcolour[3 * segId + 2][p[3 * x + 2]] += 1;
            }
        }
    }

    const int nBaseFeatures = (int)responses.size();
    const int nLocalFeatures = 2 * nBaseFeatures + 6;
#if 0
    const int nGridFeatures = 0;
    const int nGlobalFeatures = 0;
    const int nQuadraticFeatures = 0;
#else
    const int nGridFeatures = 10 * nBaseFeatures;
    const int nGlobalFeatures = 2 * nBaseFeatures;
    const int nQuadraticFeatures = 0;
#endif

    const int nTotalFeatures = nLocalFeatures + nGridFeatures +
        nGlobalFeatures + nQuadraticFeatures;

    for (unsigned i = 0; i < _nodes.size(); i++) {
        _nodes[i].features = VectorXf::Zero(nTotalFeatures);
    }

    for (unsigned y = 0; y < image.height(); y++) {
        for (unsigned x = 0; x < image.width(); x++) {
            for (int c = 0; c < image.segments().channels(); c++) {
                const int segId = image.segments()[c].at<int>(y, x);
                if (segId < 0) continue;

                for (int i = 0; i < nBaseFeatures; i++) {
                    const float p = responses[i].at<float>(y, x);
                    _nodes[segId].features[i] += p;
                    _nodes[segId].features[nBaseFeatures + i] += p * p;
                }
            }
        }
    }

    for (unsigned segId = 0; segId < _nodes.size(); segId++) {
        // mean
        _nodes[segId].features.head(nBaseFeatures) /= (float)image.segments().pixels(segId);

        // standard deviation
        for (int i = 0; i < nBaseFeatures; i++) {
            _nodes[segId].features(nBaseFeatures + i) =
                sqrt(std::max(_nodes[segId].features(nBaseFeatures + i) / (float)image.segments().pixels(segId) -
                        _nodes[segId].features(i) * _nodes[segId].features(i), 0.0f));
        }

        // entropy
        _nodes[segId].features(2 * nBaseFeatures) = (float)drwn::entropy(qcolour[3 * segId]);
        _nodes[segId].features(2 * nBaseFeatures + 1) = (float)drwn::entropy(qcolour[3 * segId + 1]);
        _nodes[segId].features(2 * nBaseFeatures + 2) = (float)drwn::entropy(qcolour[3 * segId + 2]);

        // size / shape
        const cv::Rect segBox = image.segments().boundingBox(segId);
        _nodes[segId].features(2 * nBaseFeatures + 3) = (float)segBox.width / (float)image.width();
        _nodes[segId].features(2 * nBaseFeatures + 4) = (float)segBox.height / (float)image.height();
        _nodes[segId].features(2 * nBaseFeatures + 5) = (float)image.segments().pixels(segId) /
            (float)(image.width() * image.height());
    }

    // add responses to filterbank so that we can easily average over rectangular regions
    drwnFilterBankResponse filterbank;
    if ((nGridFeatures != 0) || (nGlobalFeatures != 0)) {
        filterbank.addResponseImages(responses);
    }

    // append grid-localized neighbourhood (mean and standard deviation)
    if (nGridFeatures != 0) {
        for (unsigned segId = 0; segId < _nodes.size(); segId++) {
            const cv::Rect segBox = image.segments().boundingBox(segId);
            int nStartIndex = nLocalFeatures;

            // grid location above
            cv::Rect r = cv::Rect(segBox.x, segBox.y - 0.75 * segBox.height, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;

            // grid location below
            r = cv::Rect(segBox.x, segBox.y + 0.75 * segBox.height, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;

#if 1
            // grid location left
            r = cv::Rect(segBox.x - 0.75 * segBox.width, segBox.y, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;

            // grid location right
            r = cv::Rect(segBox.x + 0.75 * segBox.width, segBox.y, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;
#else
            // grid location left and right
            r = cv::Rect(segBox.x - 0.75 * segBox.width, segBox.y, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;

            nStartIndex -= 2 * nBaseFeatures;
            r = cv::Rect(segBox.x + 0.75 * segBox.width, segBox.y, segBox.width, segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) +=
                filterbank.variance(r.x, r.y, r.width, r.height).cast<float>().cwiseSqrt();
            nStartIndex += nBaseFeatures;
#endif

            // grid twice centered
            r = cv::Rect(segBox.x - 0.5 * segBox.width, segBox.y - 0.5 * segBox.height,
                2 * segBox.width, 2 * segBox.height);
            drwnTruncateRect(r, image.image());
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.mean(r.x, r.y, r.width, r.height).cast<float>();
            nStartIndex += nBaseFeatures;
            _nodes[segId].features.segment(nStartIndex, nBaseFeatures) =
                filterbank.variance(r.x, r.y, r.width, r.height).cwiseSqrt().cast<float>();
            nStartIndex += nBaseFeatures;

            DRWN_ASSERT(nStartIndex == nLocalFeatures + nGridFeatures);
        }
    }

    // append global image features
    if (nGlobalFeatures != 0) {
        VectorXf gf(nGlobalFeatures);
        gf.head(nBaseFeatures) = filterbank.mean().cast<float>();
        gf.tail(nBaseFeatures) = filterbank.variance().cwiseSqrt().cast<float>();
        const int nStartIndex = nLocalFeatures + nGridFeatures;
        for (unsigned segId = 0; segId < _nodes.size(); segId++) {
            _nodes[segId].features.segment(nStartIndex, nGlobalFeatures) = gf;
        }
    }

    // append quadratic features
    if (nQuadraticFeatures != 0) {
        const int nStartIndex = nLocalFeatures + nGridFeatures + nGlobalFeatures;
        for (unsigned segId = 0; segId < _nodes.size(); segId++) {
            _nodes[segId].features.segment(nStartIndex, nQuadraticFeatures) =
                _nodes[segId].features.head(nQuadraticFeatures).array().square();
        }
    }
}

void drwnNNGraphImage::cacheNodeLabels(const drwnNNGraphImageData& image)
{
    DRWN_ASSERT((int)_nodes.size() == image.segments().size());
    DRWN_ASSERT(((unsigned)image.labels().rows() == image.height()) &&
        ((unsigned)image.labels().cols() == image.width()));

    const int nLabels = image.labels().maxCoeff() + 1;
    if (nLabels <= 0) {
        DRWN_LOG_WARNING("image " << image.name() << " has no labels");
        for (unsigned segId = 0; segId < image.numSegments(); segId++) {
            _nodes[segId].label = -1;
        }
        return;
    }

    vector<VectorXi> labelCounts(image.numSegments(), VectorXi::Zero(nLabels));
    vector<int> unknownCounts(image.numSegments(), 0);
    for (int c = 0; c < image.segments().channels(); c++) {
        for (unsigned y = 0; y < image.height(); y++) {
            for (unsigned x = 0; x < image.width(); x++) {
                const int segId = image.segments()[c].at<int>(y, x);
                if (segId < 0) continue;
                const int lblId = image.labels()(y, x);
                if (lblId < 0) {
                    unknownCounts[segId] += 1;
                } else {
                    labelCounts[segId][lblId] += 1;
                }
            }
        }
    }

    int lbl;
    for (unsigned segId = 0; segId < image.numSegments(); segId++) {
        const int maxCount = labelCounts[segId].maxCoeff(&lbl);
        if (maxCount >= unknownCounts[segId]) {
            _nodes[segId].label = lbl;
        } else {
            _nodes[segId].label = -1;
        }
    }
}

// drwnNNGraph ---------------------------------------------------------------

unsigned int drwnNNGraph::K = 1;
bool drwnNNGraph::DO_PROPAGATE = true;
bool drwnNNGraph::DO_LOCAL = true;
bool drwnNNGraph::DO_SEARCH = true;
int drwnNNGraph::DO_RANDPROJ = 100;
bool drwnNNGraph::DO_ENRICHMENT = true;
int drwnNNGraph::DO_EXHAUSTIVE = 1;

bool drwnNNGraph::write(const char *filestem) const
{
    DRWN_FCN_TIC;

    drwnPersistentStorage storage;
    storage.open(filestem);

#if 1
    storage.clear();
    storage.defragment();
#endif

    for (unsigned i = 0; i < _images.size(); i++) {
        const string key = drwn::padString(toString(i), 6, '0');
        storage.write(key.c_str(), &_images[i]);
    }
    storage.close();

    DRWN_FCN_TOC;
    return true;
}

bool drwnNNGraph::read(const char *filestem)
{
    DRWN_FCN_TIC;

    // clear existing data
    _names.clear();
    _images.clear();

    // read new data
    drwnPersistentStorage storage;
    storage.open(filestem);

    _images.resize(storage.numRecords());
    for (unsigned i = 0; i < (unsigned)storage.numRecords(); i++) {
        const string key = drwn::padString(toString(i), 6, '0');
        storage.read(key.c_str(), &_images[i]);
        _names.insert(make_pair(_images[i].name(), i));
    }
    storage.close();

    DRWN_FCN_TOC;
    return true;
}

drwnNNGraph drwnNNGraph::clone(bool bWithFeatures) const
{
    drwnNNGraph g;

    g._images.reserve(_images.size());
    for (unsigned i = 0; i < _images.size(); i++) {
        g._images.push_back(_images[i].clone(bWithFeatures));
    }
    g._names = _names;

    return g;
}

size_t drwnNNGraph::numNodes() const
{
    size_t n = 0;
    for (unsigned imgIndx = 0; imgIndx < _images.size(); imgIndx++) {
        n += _images[imgIndx].numNodes();
    }
    return n;
}

size_t drwnNNGraph::numEdges() const
{
    size_t n = 0;
    for (unsigned imgIndx = 0; imgIndx < _images.size(); imgIndx++) {
        for (unsigned segId = 0; segId < _images[imgIndx].numNodes(); segId++) {
            n += _images[imgIndx][segId].edges.size();
        }
    }
    return n;
}

size_t drwnNNGraph::numNodesWithLabel(int label) const
{
    size_t n = 0;
    for (unsigned imgIndx = 0; imgIndx < _images.size(); imgIndx++) {
        for (unsigned segId = 0; segId < _images[imgIndx].numNodes(); segId++) {
            if (_images[imgIndx][segId].label == label) {
                n += 1;
            }
        }
    }
    return n;
}

int drwnNNGraph::findImage(const string& baseName) const
{
    map<string, unsigned>::const_iterator it = _names.find(baseName);
    if (it == _names.end()) return -1;
    return (int)it->second;
}

int drwnNNGraph::appendImage(const string& baseName, unsigned numNodes)
{
    return appendImage(drwnNNGraphImage(baseName, numNodes));
}

int drwnNNGraph::appendImage(const drwnNNGraphImage& image)
{
    // ensure that the image is not already in the graph
    const int imgIndex = findImage(image.name());
    if (imgIndex != -1) {
        DRWN_LOG_ERROR(image.name() << " already exists in the graph");
        return imgIndex;
    }

    _names.insert(make_pair(image.name(), (unsigned)_images.size()));
    _images.push_back(image);

    DRWN_LOG_STATUS("..." << image.name() << " with " << _images.back().numNodes() << " nodes");

    return (int)_images.size() - 1;
}

int drwnNNGraph::removeImage(unsigned imgIndx)
{
    DRWN_TODO;
    return -1;
}

pair<double, double> drwnNNGraph::energy() const
{
    DRWN_FCN_TIC;
    double totalEnergy = 0.0;
    double bestEnergy = 0.0;

    for (unsigned imgIndx = 0; imgIndx < _images.size(); imgIndx++) {
        for (unsigned segId = 0; segId < _images[imgIndx].numNodes(); segId++) {
            const drwnNNGraphEdgeList& e = _images[imgIndx][segId].edges;
            if (e.empty()) continue;
            bestEnergy += e.front().weight;
            for (drwnNNGraphEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                totalEnergy += kt->weight;
            }
        }
    }

    DRWN_FCN_TOC;
    return make_pair(totalEnergy, bestEnergy);
}

// drwnNNGraphConfig --------------------------------------------------------
//! \addtogroup drwnConfigSettings
//! \section drwnNNGraph
//! \b K               :: matches per node (default: 1)\n
//! \b propagateMove   :: execute propagate moves (default: true)\n
//! \b searchMove      :: execute random search moves (default: true)\n
//! \b localMove       :: execute local neighbourhood moves (default: true)\n
//! \b randProjMove    :: execute random projection move to horizon n (default: 100)\n
//! \b enrichmentMove  :: execute enrichment moves (default: true)\n
//! \b randExhaustive  :: exhaustive search on n random nodes per iteration (default: 1)\n
//!
//! \b imgDir          :: directory for loading image (default: data/images)\n
//! \b imgExt          :: extension of images (default: .jpg)\n
//! \b lblDir          :: directory for loading labels (default: data/labels)\n
//! \b lblExt          :: extension of labels (default: .txt)\n
//! \b segDir          :: directory for loading regions (default: data/regions)\n
//! \b segExt          :: extension of regions (default: .bin)\n

class drwnNNGraphConfig : public drwnConfigurableModule {
public:
    drwnNNGraphConfig() : drwnConfigurableModule("drwnNNGraph") { }
    ~drwnNNGraphConfig() { }

    void usage(ostream &os) const {
        os << "      K                :: matches per node (default: "
           << drwnNNGraph::K << ")\n";
        os << "      propagateMove    :: execute propagate moves (default: "
           << (drwnNNGraph::DO_PROPAGATE ? "yes" : "no") << ")\n";
        os << "      searchMove       :: execute random search moves (default: "
           << (drwnNNGraph::DO_SEARCH ? "yes" : "no") << ")\n";
        os << "      localMove        :: execute local neighbourhood moves (default: "
           << (drwnNNGraph::DO_LOCAL ? "yes" : "no") << ")\n";
        os << "      randProjMove     :: execute random projection move to horizon n (default: "
           << (drwnNNGraph::DO_RANDPROJ) << ")\n";
        os << "      enrichmentMove   :: execute enrichment moves (default: "
           << (drwnNNGraph::DO_ENRICHMENT ? "yes" : "no") << ")\n";
        os << "      randExhaustive   :: exhaustive search on a random nodes per iteration (default: "
           << drwnNNGraph::DO_EXHAUSTIVE << ")\n";

        os << "      imgDir           :: directory for loading images (default: "
           << drwnNNGraphImageData::imgDir << ")\n";
        os << "      imgExt           :: extension for loading images (default: "
           << drwnNNGraphImageData::imgExt << ")\n";
        os << "      lblDir           :: directory for loading labels (default: "
           << drwnNNGraphImageData::lblDir << ")\n";
        os << "      lblExt           :: extension for loading labels (default: "
           << drwnNNGraphImageData::lblExt << ")\n";
        os << "      segDir           :: directory for loading regions (default: "
           << drwnNNGraphImageData::segDir << ")\n";
        os << "      segExt           :: extension for loading regions (default: "
           << drwnNNGraphImageData::segExt << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "K")) {
            drwnNNGraph::K = std::max(1, atoi(value));
        } else if (!strcmp(name, "propagateMove")) {
            drwnNNGraph::DO_PROPAGATE = drwn::trueString(string(value));
        } else if (!strcmp(name, "searchMove")) {
            drwnNNGraph::DO_SEARCH = drwn::trueString(string(value));
        } else if (!strcmp(name, "localMove")) {
            drwnNNGraph::DO_LOCAL = drwn::trueString(string(value));
        } else if (!strcmp(name, "randProjMove")) {
            drwnNNGraph::DO_RANDPROJ = atoi(value);
        } else if (!strcmp(name, "enrichmentMove")) {
            drwnNNGraph::DO_ENRICHMENT = drwn::trueString(string(value));
        } else if (!strcmp(name, "randExhaustive")) {
            drwnNNGraph::DO_EXHAUSTIVE = atoi(value);
        } else if (!strcmp(name, "imgDir")) {
            drwnNNGraphImageData::imgDir = string(value);
        } else if (!strcmp(name, "imgExt")) {
            drwnNNGraphImageData::imgExt = string(value);
        } else if (!strcmp(name, "lblDir")) {
            drwnNNGraphImageData::lblDir = string(value);
        } else if (!strcmp(name, "lblExt")) {
            drwnNNGraphImageData::lblExt = string(value);
        } else if (!strcmp(name, "segDir")) {
            drwnNNGraphImageData::segDir = string(value);
        } else if (!strcmp(name, "segExt")) {
            drwnNNGraphImageData::segExt = string(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnNNGraphConfig gNNGraphConfig;

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPartsModel.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;

// drwnPartsAssignment ------------------------------------------------------

drwnPartsAssignment::drwnPartsAssignment() :
    model(NULL), centroid(0, 0), score(0.0)
{
    // do nothing
}

drwnPartsAssignment::drwnPartsAssignment(unsigned numParts) :
    model(NULL), centroid(0, 0), score(0.0)
{
    resize(numParts);
}

drwnPartsAssignment::drwnPartsAssignment(const drwnPartsAssignment& assignment) :
    model(assignment.model), centroid(assignment.centroid), locations(assignment.locations),
    occluded(assignment.occluded), score(assignment.score)
{
    // do nothing
}

drwnPartsAssignment::~drwnPartsAssignment()
{
    // do nothing
}

int drwnPartsAssignment::numEqual(const drwnPartsAssignment& assignment) const
{
    DRWN_ASSERT_MSG(assignment.locations.size() == locations.size(),
        assignment.locations.size() << " != " << locations.size());
    DRWN_ASSERT(assignment.occluded.size() == occluded.size());
    int count = 0;
    if ((centroid.y == assignment.centroid.y) &&
        (centroid.x == assignment.centroid.x))
        count += 1;
    for (unsigned i = 0; i < locations.size(); i++) {
        if ((locations[i].y == assignment.locations[i].y) &&
            (locations[i].x == assignment.locations[i].x))
            count += 1;
    }
    for (unsigned i = 0; i < occluded.size(); i++) {
        if (occluded[i] == assignment.occluded[i])
            count += 1;
    }
    return count;
}

void drwnPartsAssignment::clear()
{
    model = NULL;
    centroid = cv::Point(0, 0);
    locations.clear();
    occluded.clear();
    score = 0.0;
}

void drwnPartsAssignment::resize(unsigned numParts)
{
    locations.resize(numParts);
    occluded.resize(numParts, true);
}

void drwnPartsAssignment::print(std::ostream& os) const
{
    os << "c = " << toString(centroid) << ", x = { ";
    for (unsigned i = 0; i < locations.size(); i++) {
        os << "(" << locations[i].x << ", " << locations[i].y;
        os << ", " << (occluded[i] ? "1" : "0") << ") ";
    }
    os << "} " << score;
}

drwnPartsAssignment& drwnPartsAssignment::operator=(const drwnPartsAssignment& assignment) {
    if (&assignment != this) {
        model = assignment.model;
        centroid = assignment.centroid;
        locations = assignment.locations;
        occluded = assignment.occluded;
        score = assignment.score;
    }
    return *this;
}

bool drwnPartsAssignment::operator==(const drwnPartsAssignment& assignment) const
{
    if ((centroid.y != assignment.centroid.y) ||
        (centroid.x != assignment.centroid.x))
        return false;
    if (locations.size() != assignment.locations.size()) return false;
    for (unsigned i = 0; i < locations.size(); i++) {
        if ((locations[i].y != assignment.locations[i].y) ||
            (locations[i].x != assignment.locations[i].x))
            return false;
    }
    if (occluded.size() != assignment.occluded.size()) return false;
    for (unsigned i = 0; i < occluded.size(); i++) {
        if (occluded[i] != assignment.occluded[i])
            return false;
    }
    return true;
}

// drwnDeformationCost ------------------------------------------------------
// Structure for holding dx, dy, dx^2 and dy^2 deformation costs.

bool drwnDeformationCost::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "dx", toString(dx).c_str(), false);
    drwnAddXMLAttribute(xml, "dy", toString(dy).c_str(), false);
    drwnAddXMLAttribute(xml, "dx2", toString(dx2).c_str(), false);
    drwnAddXMLAttribute(xml, "dy2", toString(dy2).c_str(), false);

    return true;
}

bool drwnDeformationCost::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "dx") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "dy") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "dx2") != NULL);
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "dy2") != NULL);

    dx = atoi(drwnGetXMLAttribute(xml, "dx"));
    dy = atoi(drwnGetXMLAttribute(xml, "dy"));
    dx2 = atoi(drwnGetXMLAttribute(xml, "dx2"));
    dy2 = atoi(drwnGetXMLAttribute(xml, "dy2"));

    return true;
}

// drwnPartsInference -------------------------------------------------------

int drwnPartsInference::MESSAGE_PASSING_SCALE = 1; // no rescaling
double drwnPartsInference::DEFAULT_LAMBDA = 5.0e-3;
double drwnPartsInference::DEFAULT_OCCLUSION_COST = 0.0; // actually a reward

drwnPartsInference::drwnPartsInference(int numParts, int width, int height) :
    _lambda(DEFAULT_LAMBDA), _occlusionCost(DEFAULT_OCCLUSION_COST),
    _width(width), _height(height), _nParts(numParts)
{
    // allocate memory
    _centroidPrior = cv::Mat::zeros(_height, _width, CV_32FC1);
    for (int i = 0; i < _nParts; i++) {
        _partCosts.push_back(make_pair(_centroidPrior.clone(), _centroidPrior.clone()));
    }
    _partOffsets.resize(_nParts, cv::Point(0, 0));
    _pairwiseCosts.resize(_nParts);
}

drwnPartsInference::~drwnPartsInference()
{
    // do nothing
}

// define problem
void drwnPartsInference::setCentroidPrior(const cv::Mat& centroidPrior)
{
    if (centroidPrior.data == NULL) {
        _centroidPrior.setTo(cv::Scalar::all(0));
    } else {
        DRWN_ASSERT((centroidPrior.rows == _height) && (centroidPrior.cols == _width));
        centroidPrior.copyTo(_centroidPrior);
    }
}

void drwnPartsInference::setPartCosts(int partId, const cv::Mat& matchingCost,
    const cv::Point& offset, const drwnDeformationCost& pairwiseCost, const cv::Mat& occlusionCost)
{
    DRWN_ASSERT((partId >= 0) && (partId < _nParts));
    DRWN_ASSERT(matchingCost.data != NULL);
    DRWN_ASSERT_MSG((matchingCost.rows == _height) && (matchingCost.cols == _width),
        "(" << matchingCost.rows << " == " << _height << ") && ("
        << matchingCost.cols << " == " << _width << ")");
    DRWN_ASSERT((occlusionCost.data == NULL) || ((occlusionCost.rows == _height) && (occlusionCost.cols == _width)));

    _partOffsets[partId] = offset;

    // NOTE: beliefs of parts are offset to account for offset to centroid (making
    // the message passing calculations easier)
    _partCosts[partId].first = drwnTranslateMatrix(matchingCost, _partOffsets[partId], 0.0);
    if (occlusionCost.data != NULL) {
        _partCosts[partId].second = drwnTranslateMatrix(occlusionCost, _partOffsets[partId], _occlusionCost);
    } else {
        _partCosts[partId].second.setTo(cv::Scalar::all(_occlusionCost));
    }

    _pairwiseCosts[partId] = pairwiseCost;
}

drwnPartsAssignment drwnPartsInference::inference() const
{
    DRWN_FCN_TIC;

    // initialize beliefs (zero)
    cv::Mat beliefs_c(_centroidPrior.clone());
    vector<cv::Mat> beliefs_x(_nParts);
    vector<cv::Mat> beliefs_z(_nParts);
    for (int i = 0; i < _nParts; i++) {
        beliefs_x[i] = cv::Mat::zeros(_height, _width, CV_32FC1);
        beliefs_z[i] = cv::Mat::zeros(2, 1, CV_32FC1);
    }

    // begin message passing
    vector<cv::Mat> messages_zx;
    vector<cv::Mat> messages_xc;

    // messages from z (occlusions) to x (location)
    for (int i = 0; i < _nParts; i++) {
        // compute message
        cv::Mat h(_height, _width, CV_32FC1);
        cv::max(_partCosts[i].first, _partCosts[i].second, h);
        messages_zx.push_back(h);

        // absorb into belief
        cv::add(beliefs_x[i], messages_zx[i], beliefs_x[i]);
    }

    // messages from x (location) to x_c (centroid)
    for (int i = 0; i < _nParts; i++) {
        // compute forward message
        messages_xc.push_back(computeLocationMessage(beliefs_x[i], _pairwiseCosts[i], cv::Mat()));

        // absorb message into belief state
        cv::add(beliefs_c, messages_xc[i], beliefs_c);
    }

    // messages from x_c to x
    for (int i = 0; i < _nParts; i++) {
        // compute backward message
        cv::Mat h = computeLocationMessage(beliefs_c, _pairwiseCosts[i], messages_xc[i]);

        // absorb message into belief state
        cv::add(beliefs_x[i], h, beliefs_x[i]);
    }

    // messages from x to z
    for (int i = 0; i < _nParts; i++) {
        cv::Mat h(_height, _width, CV_32FC1);
        cv::subtract(beliefs_x[i], messages_zx[i], h);

        cv::Mat m(_height, _width, CV_32FC1);
        cv::add(h, _partCosts[i].first, m);

        double minVal, maxVal;
        cv::minMaxLoc(m, &minVal, &maxVal);
        beliefs_z[i].at<float>(0, 0) += (float)maxVal;

        cv::add(h, _partCosts[i].second, m);
        cv::minMaxLoc(m, &minVal, &maxVal);
        beliefs_z[i].at<float>(1, 0) += (float)maxVal;
    }

    // decode beliefs (shifting locations back by offset)
    double energy = DRWN_DBL_MAX;
    drwnPartsAssignment mapAssignment(_nParts);
    //mapAssignment.model = this;

    for (int i = 0; i < _nParts; i++) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(beliefs_x[i], &minVal, &maxVal, &minLoc, &maxLoc);
        mapAssignment.locations[i] = maxLoc;

        mapAssignment.locations[i].x += _partOffsets[i].x;
        mapAssignment.locations[i].y += _partOffsets[i].y;

        mapAssignment.occluded[i] = (beliefs_z[i].at<float>(1, 0) > beliefs_z[i].at<float>(0, 0));

        DRWN_LOG_DEBUG("x[" << i << "] = " << toString(maxLoc)
            << ", z[" << i << "] = " << (mapAssignment.occluded[i] ? 1 : 0)
            << ", E(x, z) = " << maxVal);
    }

    // centroid
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(beliefs_c, &minVal, &maxVal, &minLoc, &maxLoc);
    mapAssignment.centroid = maxLoc;

    DRWN_LOG_DEBUG("c = " << toString(maxLoc) << ", E(c, x, z) = " << maxVal);

    energy = -maxVal;
    mapAssignment.score = energy;

    DRWN_FCN_TOC;
    return mapAssignment;
}

drwnPartsAssignment drwnPartsInference::inference(const cv::Point& centroid) const
{
    DRWN_FCN_TIC;

    drwnPartsAssignment mapAssignment(_nParts);
    mapAssignment.centroid = centroid;
    mapAssignment.score = 0.0;

    cv::Mat c(_height, _width, CV_32FC1, cv::Scalar(DRWN_EPSILON - DRWN_FLT_MAX));
    c.at<float>(centroid.y, centroid.x) = 0.0;

    // maximize each part independently
    cv::Mat m(_height, _width, CV_32FC1);
    for (int i = 0; i < _nParts; i++) {
        // matching cost
        cv::max(_partCosts[i].first, _partCosts[i].second, m);

        // deformation cost
        cv::Mat d = computeLocationMessage(c, _pairwiseCosts[i], cv::Mat());

        // total cost
        cv::add(m, d, d);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(d, &minVal, &maxVal, &minLoc, &maxLoc);
        mapAssignment.locations[i] = maxLoc;
        mapAssignment.occluded[i] = (m.at<float>(maxLoc.y, maxLoc.x) !=
            _partCosts[i].first.at<float>(maxLoc.y, maxLoc.x));
        mapAssignment.locations[i].x += _partOffsets[i].x;
        mapAssignment.locations[i].y += _partOffsets[i].y;

        // accumulate energy
        mapAssignment.score -= maxVal;
    }

    DRWN_FCN_TOC;
    return mapAssignment;
}

cv::Mat drwnPartsInference::energyLandscape() const
{
    DRWN_FCN_TIC;
    cv::Mat energy(cv::Mat::zeros(_height, _width, CV_32FC1));

    cv::Mat m(_height, _width, CV_32FC1);

    // maximize each part independently
    for (int i = 0; i < _nParts; i++) {

        // matching cost
        cv::max(_partCosts[i].first, _partCosts[i].second, m);

        cv::Mat d = computeLocationMessage(m, _pairwiseCosts[i], cv::Mat());
        cv::subtract(energy, d, energy);
    }

    DRWN_FCN_TOC;
    return energy;
}

// inference message passing between location variables
// (max-product version)
cv::Mat drwnPartsInference::computeLocationMessage(const cv::Mat& belief,
    const drwnDeformationCost& dcost, const cv::Mat& msgIn) const
{
    // subtract incoming message from belief state
    cv::Mat h(belief.rows, belief.cols, CV_32FC1);
    if (msgIn.data != NULL) {
        cv::subtract(belief, msgIn, h);
    } else {
        belief.copyTo(h);
    }

    // rescale images for faster message passing
    const double linScale = std::max(1.0, (double)MESSAGE_PASSING_SCALE);
    const double quadScale = linScale * linScale;
    if (MESSAGE_PASSING_SCALE > 1) {
        drwnResizeInPlace(h, h.rows / MESSAGE_PASSING_SCALE,
            h.cols / MESSAGE_PASSING_SCALE, CV_INTER_LINEAR);
    }

    // x-dimension linear cost
    if (dcost.dx != 0.0) {
        const float nu = linScale * _lambda * dcost.dx;
        for (int y = 0; y < h.rows; y++) {
            float *p = h.ptr<float>(y);
            for (int x = 1; x < h.cols; x++) {
                p[x] = std::max(p[x], p[x - 1] - nu);
            }
            for (int x = h.cols - 2; x >= 0; x--) {
                p[x] = std::max(p[x], p[x + 1] - nu);
            }
        }
    }

    // y-dimension linear cost
    if (dcost.dy != 0.0) {
        const float nu = linScale * _lambda * dcost.dy;
        for (int x = 0; x < h.cols; x++) {
            float *p = h.ptr<float>(0) + x;
            for (int y = 1; y < h.rows; y++) {
                p[y * h.cols] = std::max(p[y * h.cols], p[(y - 1) * h.cols] - nu);
            }
            for (int y = h.rows - 2; y >= 0; y--) {
                p[y * h.cols] = std::max(p[y * h.cols], p[(y + 1) * h.cols] - nu);
            }
        }
    }

    // x-dimension quadratic cost
    if (dcost.dx2 != 0.0) {
        const float nu = quadScale * _lambda * dcost.dx2;

        vector<int> v(h.cols);
        vector<float> z(h.cols + 1); // TODO: can this be int?
        vector<float> d(h.cols);

        for (int y = 0; y < h.rows; y++) {
            float *p = h.ptr<float>(y);
            int k = 0;
            v[0] = 0; //fill(v.begin(), v.end(), 0);
            fill(z.begin(), z.end(), DRWN_FLT_MAX);
            z[0] = -DRWN_FLT_MAX;

            for (int x = 1; x < h.cols; x++) {
                float s;
                while (1) {
                    DRWN_ASSERT(k >= 0);
                    s = ((p[x] - nu * x*x) - (p[v[k]] - nu * v[k]*v[k])) /
                        (float)(-2.0 * nu * (x - v[k]));
                    if (s <= z[k]) {
                        k -= 1;
                    } else {
                        break;
                    }
                }

                k += 1;
                v[k] = x;
                z[k] = s;
            }

            k = 0;
            for (int x = 0; x < h.cols; x++) {
                while (z[k + 1] < (float)x) {
                    k += 1;
                }
                d[x] = p[v[k]] - nu * (x - v[k]) * (x - v[k]);
            }

            /*
            for (int x = 0; x < h->cols; x++) {
                p[x] = d[x];
            }
            */
            memcpy(&p[0], &d[0], h.cols * sizeof(float));
        }
    }

    // y-dimension quadratic cost
    if (dcost.dy2 != 0.0) {
        const float nu = quadScale * _lambda * dcost.dy2;

        vector<int> v(h.rows);
        vector<float> z(h.rows + 1); // TODO: can this be int?
        vector<float> d(h.rows);

        for (int x = 0; x < h.cols; x++) {
            float *p = h.ptr<float>(0) + x;
            int k = 0;
            v[0] = 0; //fill(v.begin(), v.end(), 0);
            fill(z.begin(), z.end(), DRWN_FLT_MAX);
            z[0] = -DRWN_FLT_MAX;

            for (int y = 1; y < h.rows; y++) {
                float s;
                while (1) {
                    DRWN_ASSERT(k >= 0);
                    s = ((p[y * h.cols] - nu * y*y) - (p[v[k] * h.cols] - nu * v[k]*v[k])) /
                        (float)(-2.0 * nu * (y - v[k]));
                    if (s <= z[k]) {
                        k -= 1;
                    } else {
                        break;
                    }
                }

                k += 1;
                v[k] = y;
                z[k] = s;
            }

            k = 0;
            for (int y = 0; y < h.rows; y++) {
                while (z[k + 1] < (float)y) {
                    k += 1;
                }
                d[y] = p[v[k] * h.cols] - nu * (y - v[k]) * (y - v[k]);
            }

            for (int y = 0; y < h.rows; y++) {
                p[y * h.cols] = d[y];
            }
        }
    }

    // scale messages back to full size
    if (MESSAGE_PASSING_SCALE > 1) {
        drwnResizeInPlace(h, belief.rows, belief.cols, CV_INTER_LINEAR);
    }

    return h;
}

// drwnPart -----------------------------------------------------------------

int drwnPart::MATCH_MODE = CV_TM_CCOEFF_NORMED;

drwnPart::drwnPart() {
    _extent = cvSize(0, 0);
    _offset = cvPoint(0, 0);
}

drwnPart::drwnPart(const cv::Size& extent, unsigned channels) :
    _extent(extent), _offset(0, 0)
{
    DRWN_ASSERT(channels > 0);
    _weights.resize(channels, cv::Mat());
}

drwnPart::drwnPart(const drwnPart& part) :
    _extent(part._extent), _offset(part._offset), _dcost(part._dcost)
{
    _weights.reserve(part._weights.size());
    for (unsigned i = 0; i < part._weights.size(); i++) {
        _weights.push_back(part._weights[i].clone());
    }
}

drwnPart::~drwnPart()
{
    // do nothing
}

// i/o
void drwnPart::clear()
{
    _extent = cv::Size(0, 0);
    _weights.clear();
    _offset = cv::Point(0, 0);
    _dcost = drwnDeformationCost();
}

bool drwnPart::save(drwnXMLNode& xml) const
{
    // write offset and extent
    drwnAddXMLAttribute(xml, "x", toString(_offset.x).c_str(), false);
    drwnAddXMLAttribute(xml, "y", toString(_offset.y).c_str(), false);
    drwnAddXMLAttribute(xml, "w", toString(_extent.width).c_str(), false);
    drwnAddXMLAttribute(xml, "h", toString(_extent.height).c_str(), false);

    // write channel feature weights
    for (unsigned i = 0; i < _weights.size(); i++) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "weight", NULL, false);
        drwnAddXMLAttribute(*node, "rows", toString(_weights[i].rows).c_str(), false);
        drwnAddXMLAttribute(*node, "cols", toString(_weights[i].cols).c_str(), false);
        drwnXMLUtils::serialize(*node, (const char *)_weights[i].data,
            _weights[i].cols * _weights[i].rows * sizeof(float));
    }

    // write deformation costs
    drwnXMLNode *node = drwnAddXMLChildNode(xml, _dcost.type(), NULL);
    _dcost.save(*node);

    return true;
}

bool drwnPart::load(drwnXMLNode& xml)
{
    // read offset and extent
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "x") != NULL);
    _offset.x = atoi(drwnGetXMLAttribute(xml, "x"));
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "y") != NULL);
    _offset.y = atoi(drwnGetXMLAttribute(xml, "y"));

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "w") != NULL);
    _extent.width = atoi(drwnGetXMLAttribute(xml, "w"));
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "h") != NULL);
    _extent.height = atoi(drwnGetXMLAttribute(xml, "h"));

    // read channel feature weights
    for (drwnXMLNode *node = xml.first_node("weight"); node != NULL; node = node->next_sibling("weight")) {
        DRWN_ASSERT(!drwnIsXMLEmpty(*node));

        const int r = atoi(drwnGetXMLAttribute(*node, "rows"));
        const int c = atoi(drwnGetXMLAttribute(*node, "cols"));
        _weights.push_back(cv::Mat(r, c, CV_32FC1));
        drwnXMLUtils::deserialize(*node, (char *)_weights.back().data,
            r * c * sizeof(float));
    }

    // read deformation costs
    drwnXMLNode *node = xml.first_node(_dcost.type());
    DRWN_ASSERT((node != NULL) && !drwnIsXMLEmpty(*node));
    _dcost.load(*node);

    return true;
}

void drwnPart::swap(drwnPart& part)
{
    std::swap(_extent, part._extent);
    _weights.swap(part._weights);
    std::swap(_offset, part._offset);
    std::swap(_dcost, part._dcost);
}

// learning
void drwnPart::setWeights(const cv::Mat& weights)
{
    DRWN_ASSERT((weights.data != NULL) && (_weights.size() == 1));
    _weights[0] = weights.clone();
}

void drwnPart::setWeights(const vector<cv::Mat>& weights)
{
    DRWN_ASSERT(weights.size() == _weights.size());

    for (int i = 0; i < (int)weights.size(); i++) {
        _weights[i] = weights[i].clone();
        DRWN_ASSERT((_weights[i].rows == _weights[0].rows) &&
            (_weights[i].cols == _weights[0].cols));
    }
}

// inference
cv::Mat drwnPart::unaryCosts(const cv::Mat& features) const
{
    DRWN_ASSERT(_weights.size() == 1);
    DRWN_ASSERT(features.data != NULL);
    DRWN_FCN_TIC;

    cv::Mat response = cv::Mat::zeros(features.rows, features.cols, CV_32FC1);

    cv::Mat subRect = response(cv::Rect(0, 0,
            features.cols - _weights[0].cols + 1,
            features.rows - _weights[0].rows + 1));
    cv::matchTemplate(features, _weights[0], subRect, drwnPart::MATCH_MODE);

    DRWN_FCN_TOC;
    return response;
}

cv::Mat drwnPart::unaryCosts(const vector<cv::Mat>& features) const
{
    DRWN_ASSERT(features.size() == _weights.size());
    DRWN_FCN_TIC;

    const int H = features.front().rows;
    const int W = features.front().cols;
    const int h = _weights.front().rows;
    const int w = _weights.front().cols;
    //DRWN_ASSERT_MSG((H >= h) && (W >= w), "(W,H) = (" << W << ", " << H << "); (w, h) = ("
    //    << w << ", " << h << ")");

    cv::Mat response = cv::Mat::zeros(H, W, CV_32FC1);
    if ((W < w) || (H < h)) {
        //DRWN_LOG_ERROR("image is smaller than template (W,H) = (" << W << ", "
        //    << H << "); (w, h) = (" << w << ", " << h << "). Shifted too far?");
        DRWN_FCN_TOC;
        return response;
    }

    cv::Mat subRect = response(cv::Rect(0, 0, W - w + 1, H - h + 1));

    // sum response for each channel
    cv::Mat channelResp(H - h + 1, W - w + 1, CV_32FC1);
    for (unsigned i = 0; i < _weights.size(); i++) {
        cv::matchTemplate(features[i], _weights[i], channelResp, drwnPart::MATCH_MODE);
        cv::add(channelResp, subRect, subRect);
    }

    subRect *= 1.0 / (double)_weights.size();

    DRWN_FCN_TOC;
    return response;
}

double drwnPart::pairwiseCost(const cv::Point& x, const cv::Point& c) const
{
    return _dcost.cost(cv::Point(x.x - _offset.x, x.y - _offset.y), c);
}

// operators
drwnPart& drwnPart::operator=(const drwnPart& part)
{
    if (&part != this) {
        drwnPart tmp(part);
        this->swap(tmp);
    }

    return *this;
}

// friends (overlap in pixel space)
double overlap(const drwnPart& partA, const cv::Point& locationA,
    const drwnPart& partB, const cv::Point& locationB)
{
    // define part regions
    cv::Rect rA(locationA.x, locationA.y, partA._extent.width, partA._extent.height);
    cv::Rect rB(locationB.x, locationB.y, partB._extent.width, partB._extent.height);

    // find intersection
    int ix = std::max(rA.x, rB.x);
    int iy = std::max(rA.y, rB.y);
    int iw = std::min(rA.x + rA.width, rB.x + rB.width) - ix;
    int ih = std::min(rA.y + rA.height, rB.y + rB.height) - iy;

    if ((iw < 0) || (ih < 0))
        return 0.0;

    // intersection over union
    double iou = (double)(iw * ih) /
        (double)(rA.width * rA.height + rB.width * rB.height - iw * ih);
    DRWN_LOG_DEBUG("iou: " << iou << ": " << toString(rA) << " <> " << toString(rB));
    return iou;
}

// drwnPartsModel -----------------------------------------------------------

drwnPartsModel::drwnPartsModel()
{
    _baseSize = cvSize(1, 1);
}

drwnPartsModel::drwnPartsModel(const drwnPartsModel& model) :
    _baseSize(model._baseSize), _parts(model._parts)
{
    // copy parts
    _parts.reserve(model._parts.size());
    for (unsigned i = 0; i < model._parts.size(); i++) {
        _parts.push_back(new drwnPart(*model._parts[i]));
    }
}

drwnPartsModel::~drwnPartsModel()
{
    // delete parts
    for (unsigned i = 0; i < _parts.size(); i++) {
        delete _parts[i];
    }
}

void drwnPartsModel::clear()
{
    _baseSize = cv::Size(1, 1);
    for (unsigned i = 0; i < _parts.size(); i++) {
        delete _parts[i];
    }
    _parts.clear();
}

bool drwnPartsModel::save(drwnXMLNode &xml) const
{
    drwnAddXMLAttribute(xml, "w", toString(_baseSize.width).c_str(), false);
    drwnAddXMLAttribute(xml, "h", toString(_baseSize.height).c_str(), false);

    for (unsigned i = 0; i < _parts.size(); i++) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnPart", NULL, false);
        _parts[i]->save(*node);
    }

    return true;
}

bool drwnPartsModel::load(drwnXMLNode& xml)
{
    clear();

    DRWN_ASSERT(drwnGetXMLAttribute(xml, "w") != NULL);
    _baseSize.width = atoi(drwnGetXMLAttribute(xml, "w"));
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "h") != NULL);
    _baseSize.height = atoi(drwnGetXMLAttribute(xml, "h"));

    const int numChildren = drwnCountXMLChildren(xml, "drwnPart");
    _parts.reserve(numChildren);
    for (drwnXMLNode *node = xml.first_node("drwnPart"); node != NULL; node = node->next_sibling("drwnPart")) {
        DRWN_ASSERT(!drwnIsXMLEmpty(*node));
        _parts.push_back(new drwnPart());
        _parts.back()->load(*node);
    }

    return true;
}

void drwnPartsModel::swap(drwnPartsModel& model)
{
    _parts.swap(model._parts);
}

// inference
double drwnPartsModel::inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment) const
{
    DRWN_ASSERT(img.data != NULL);

    // compute unary matching costs
    vector<cv::Mat> mcosts = computeMatchingCosts(img);

    // run inference
    drwnPartsInference infObject(_parts.size(), img.cols, img.rows);
    for (unsigned i = 0; i < _parts.size(); i++) {
        infObject.setPartCosts(i, mcosts[i], _parts[i]->getOffset(), _parts[i]->getDCosts());
    }

    mapAssignment = infObject.inference();
    mapAssignment.model = this;

    return mapAssignment.score;
}

double drwnPartsModel::inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
    const cv::Point &centroidPrior) const
{
    DRWN_ASSERT(img.data != NULL);

    // compute unary matching costs
    vector<cv::Mat> mcosts = computeMatchingCosts(img);

    // run inference
    drwnPartsInference infObject(_parts.size(), img.cols, img.rows);
    for (unsigned i = 0; i < _parts.size(); i++) {
        infObject.setPartCosts(i, mcosts[i], _parts[i]->getOffset(), _parts[i]->getDCosts());
    }

    mapAssignment = infObject.inference(centroidPrior);
    mapAssignment.model = this;

    return mapAssignment.score;
}

double drwnPartsModel::inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
    const vector<cv::Mat>& partPriors, const cv::Mat& centroidPrior) const
{
    DRWN_ASSERT(img.data != NULL);
    DRWN_ASSERT(partPriors.empty() || (partPriors.size() == _parts.size()));

    // compute unary matching costs
    vector<cv::Mat> mcosts = computeMatchingCosts(img);

    // set unary and pariwise costs
    drwnPartsInference infObject(_parts.size(), img.cols, img.rows);
    infObject.setCentroidPrior(centroidPrior);

    if (partPriors.empty()) {
        for (unsigned i = 0; i < _parts.size(); i++) {
            infObject.setPartCosts(i, mcosts[i], _parts[i]->getOffset(), _parts[i]->getDCosts());
        }
    } else {
        for (unsigned i = 0; i < _parts.size(); i++) {
            cv::Mat ocost(img.rows, img.cols, CV_32FC1, cv::Scalar(drwnPartsInference::DEFAULT_OCCLUSION_COST));

            if (partPriors[i].data != NULL) {
                cv::add(mcosts[i], partPriors[i], mcosts[i]);
                cv::add(ocost, partPriors[i], ocost);
            }

            infObject.setPartCosts(i, mcosts[i], _parts[i]->getOffset(), _parts[i]->getDCosts(), ocost);
        }
    }

    // run inference
    mapAssignment = infObject.inference();
    mapAssignment.model = this;

    return mapAssignment.score;
}

double drwnPartsModel::inference(const cv::Mat& img, drwnPartsAssignment& mapAssignment,
    double& bestScale, double startScale, double endScale, int numLevels) const
{
    DRWN_ASSERT(img.data != NULL);
    DRWN_ASSERT((endScale <= startScale) && (numLevels > 0));
    const double scaleFactor = exp(log(endScale / startScale) / (double)numLevels);

    bestScale = 0.0;
    double bestScore = DRWN_DBL_MAX;
    mapAssignment.clear();

    cv::Mat scaledImage = img.clone();
    drwnResizeInPlace(scaledImage, startScale * img.rows, startScale * img.cols);
    for (int i = 0; i < numLevels; i++) {
        drwnPartsAssignment assignment;
        double score = inference(scaledImage, assignment);
        if (score < bestScore) {
            bestScale = (double)img.cols / (double)scaledImage.cols;
            bestScore = score;
            mapAssignment = assignment;
        }

        if ((scaleFactor * scaledImage.rows < 1.0) ||
            (scaleFactor * scaledImage.cols < 1.0)) break;

        drwnResizeInPlace(scaledImage, scaleFactor * scaledImage.rows,
            scaleFactor * scaledImage.cols);
    }

    return bestScore;
}

cv::Mat drwnPartsModel::energyLandscape(const cv::Mat& img) const
{
    DRWN_ASSERT(img.data != NULL);

    // compute unary matching costs
    vector<cv::Mat> mcosts = computeMatchingCosts(img);

    // set unary and pariwise costs
    drwnPartsInference infObject(_parts.size(), img.cols, img.rows);
    for (unsigned i = 0; i < _parts.size(); i++) {
        infObject.setPartCosts(i, mcosts[i], _parts[i]->getOffset(), _parts[i]->getDCosts());
    }

    // run inference
    return infObject.energyLandscape();
}

void drwnPartsModel::slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections) const
{
    if ((img.rows < _baseSize.height) || (img.cols < _baseSize.width)) {
        return;
    }

    DRWN_FCN_TIC;
    //DRWN_LOG_VERBOSE("...running inference on " << toString(*img) << " for " << toString(_baseSize));
    cv::Mat energy = energyLandscape(img);

    for (int y = _baseSize.height / 2; y < energy.rows - _baseSize.height / 2; y++) {
        const float *p = energy.ptr<const float>(y);
        for (int x = _baseSize.width / 2; x < energy.cols - _baseSize.width / 2; x++) {
            if (p[x] < -0.1 * _parts.size()) {
                detections.push_back(drwnObject(cvRect(x - _baseSize.width / 2, y - _baseSize.height / 2,
                            _baseSize.width, _baseSize.height)));
                detections.back().score = -p[x];
            }
        }
    }

    DRWN_FCN_TOC;
}

void drwnPartsModel::slidingWindowDetections(const cv::Mat& img, drwnObjectList& detections,
    int numLevelsPerOctave) const
{
    DRWN_ASSERT(img.data != NULL);
    DRWN_ASSERT(numLevelsPerOctave > 0);
    const double scaleFactor = exp(-1.0 * log(2.0) / (double)numLevelsPerOctave);

    cv::Mat scaledImage = img.clone();
    while ((scaledImage.cols >= _baseSize.width) && (scaledImage.rows >= _baseSize.height)) {

        unsigned startIndx = detections.size();
        slidingWindowDetections(scaledImage, detections);
        const double rescaleFactor = (double)img.cols / (double)scaledImage.cols;
        while (startIndx < detections.size()) {
            detections[startIndx].scale(rescaleFactor);
            startIndx += 1;
        }

        drwnResizeInPlace(scaledImage, scaleFactor * scaledImage.rows,
            scaleFactor * scaledImage.cols);
    }
}

// visualization
cv::Mat drwnPartsModel::showMAPPartLocations(const cv::Mat& img) const
{
    drwnPartsAssignment mapAssignment;
    double energy = inference(img, mapAssignment);
    return showMAPPartLocations(img, mapAssignment, energy);
}

cv::Mat drwnPartsModel::showMAPPartLocations(const cv::Mat& img,
    const drwnPartsAssignment& assignment, double energy, double scale) const
{
    // allocate images
    cv::Mat grey = (img.channels() == 1) ? img.clone() : drwnGreyImage(img);
    cv::Mat canvas = drwnColorImage(grey);

    // draw object parts
    DRWN_ASSERT(assignment.numParts() == numParts());
    for (int i = 0; i < (int)_parts.size(); i++) {
        cv::Point topLeft = assignment.locations[i];
        cv::Point botRight(topLeft.x + _parts[i]->width(), topLeft.y + _parts[i]->height());
        cv::Scalar partColour = partColorMap(i);

        if (scale != 1.0) {
            topLeft.x = (int)(scale * topLeft.x);
            topLeft.y = (int)(scale * topLeft.y);
            botRight.x = (int)(scale * botRight.x);
            botRight.y = (int)(scale * botRight.y);
        }

        // shade occluded parts
        if (assignment.occluded[i]) {
            drwnShadeRectangle(canvas, cv::Rect(topLeft.x, topLeft.y,
                    botRight.x - topLeft.x, botRight.y - topLeft.y),
                partColour, 1.0, DRWN_FILL_CROSSHATCH);
        }

        // show bounding box
        drwnDrawBoundingBox(canvas, cv::Rect(topLeft.x, topLeft.y,
                botRight.x - topLeft.x, botRight.y - topLeft.y),
            partColour, CV_RGB(0, 0, 0), 2);
    }

    // show energy
    if (energy != DRWN_DBL_MAX) {
        int baseline;
	const string engStr = toString(energy);
        cv::Size textSize = cv::getTextSize(engStr, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseline);
        cv::putText(canvas, engStr.c_str(), cv::Point(2, textSize.height + 2),
            CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(127, 127, 255), 1);
    }

    return canvas;
}

cv::Mat drwnPartsModel::showPartEnergyLandscape(const cv::Mat& img) const
{
    // allocate images
    cv::Mat grey = drwnGreyImage(img);
    cv::Mat canvas = drwnColorImage(grey);

    // brighten canvas
    canvas = 0.25 * canvas + 191;

    // compute unary matching costs
    vector<cv::Mat> mcosts = computeMatchingCosts(img);

    // combine part energies
    vector<cv::Mat> views;
    for (int i = 0; i < (int)_parts.size(); i++) {
        views.push_back(canvas.clone());

        cv::Mat e = drwnTranslateMatrix(mcosts[i], _parts[i]->getOffset(), 0.0);
        drwnScaleToRange(e, 0.0, 1.0);

        drwnOverlayMask(views.back(), e, partColorMap(i), 0.75);
    }

    canvas = drwnCombineImages(views);

    return canvas;
}

// operators
drwnPartsModel& drwnPartsModel::operator=(const drwnPartsModel& model)
{
    if ((void *)&model != (void *)this) {
        _parts = model._parts;
    }

    return *this;
}

// initial part locations
vector<pair<cv::Point, cv::Size> > drwnPartsModel::initializePartLocations(int nParts,
    const cv::Size& imgSize)
{
    vector<pair<cv::Point, cv::Size> > partLocations(nParts);

    // special cases
    // TODO: fix this
    if (nParts == 1) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[0].second = imgSize;
    } else if (nParts == 2) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[1].first = cv::Point(0, imgSize.height/2);
        partLocations[0].second = cv::Size(imgSize.width, (imgSize.height + 1)/2);
        partLocations[1].second = partLocations[0].second;
    } else if (nParts == 3) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[0].second = cv::Size(imgSize.width, (imgSize.height + 1)/2);
        partLocations[1].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 1)/2);
        partLocations[2].second = partLocations[1].second;
        partLocations[1].first = cv::Point(0, imgSize.height - partLocations[1].second.height);
        partLocations[2].first = cv::Point(imgSize.width - partLocations[2].second.width,
            imgSize.height - partLocations[2].second.height);
    } else if (nParts == 4) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[1].first = cv::Point(imgSize.width/2, 0);
        partLocations[2].first = cv::Point(0, imgSize.height/2);
        partLocations[3].first = cv::Point(imgSize.width/2, imgSize.height/2);
        partLocations[0].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 1)/2);
        partLocations[1].second = partLocations[0].second;
        partLocations[2].second = partLocations[0].second;
        partLocations[3].second = partLocations[0].second;
    } else if (nParts == 5) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[1].first = cv::Point(imgSize.width/2, 0);
        partLocations[2].first = cv::Point(imgSize.width/2, imgSize.height/2);
        partLocations[3].first = cv::Point(0, imgSize.height/2);
        partLocations[4].first = cv::Point(0, 0);
        partLocations[0].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 1)/2);
        partLocations[1].second = partLocations[0].second;
        partLocations[2].second = partLocations[0].second;
        partLocations[3].second = partLocations[0].second;
        partLocations[4].second = imgSize;
    } else if (nParts == 6) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[1].first = cv::Point(imgSize.width/2, 0);
        partLocations[2].first = cv::Point(imgSize.width/2, imgSize.height/2);
        partLocations[3].first = cv::Point(0, imgSize.height/2);
        partLocations[4].first = cv::Point(imgSize.width/4, 0);
        partLocations[0].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 1)/2);
        partLocations[1].second = partLocations[0].second;
        partLocations[2].second = partLocations[0].second;
        partLocations[3].second = partLocations[0].second;
        partLocations[4].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 3)/4);
        partLocations[5].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 3)/4);
        partLocations[5].first = cv::Point(imgSize.width/4,
            imgSize.height - partLocations[5].second.height);
    } else if (nParts == 8) {
        partLocations[0].first = cv::Point(0, 0);
        partLocations[1].first = cv::Point(imgSize.width/2, 0);
        partLocations[2].first = cv::Point(imgSize.width/2, imgSize.height/2);
        partLocations[3].first = cv::Point(0, imgSize.height/2);
        partLocations[4].first = cv::Point(imgSize.width/4, 0);
        partLocations[0].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 1)/2);
        partLocations[1].second = partLocations[0].second;
        partLocations[2].second = partLocations[0].second;
        partLocations[3].second = partLocations[0].second;
        partLocations[4].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 3)/4);
        partLocations[5].second = cv::Size((imgSize.width + 1)/2, (imgSize.height + 3)/4);
        partLocations[5].first = cv::Point(imgSize.width/4,
            imgSize.height - partLocations[5].second.height);
        partLocations[6].first = cv::Point(0, imgSize.height/4);
        partLocations[6].second = cv::Size((imgSize.width + 3)/4, (imgSize.height + 1)/2);
        partLocations[7].second = cv::Size((imgSize.width + 3)/4, (imgSize.height + 1)/2);
        partLocations[7].first = cv::Point(imgSize.width - partLocations[7].second.width, imgSize.height/4);
    } else {
        // general case
        double theta = 0.0;
        cv::Size partSize((imgSize.width + 1) / 2, (imgSize.height + 1) / 2);
        for (int i = 0; i < nParts; i++) {
            partLocations[i].second = partSize;

            partLocations[i].first.x = (int)(cos(theta) * imgSize.width/4 +
                imgSize.width/2 - partSize.width/2);
            partLocations[i].first.y = (int)(sin(theta) * imgSize.height/4 +
                imgSize.height/2 - partSize.height/2);

            partLocations[i].first.x = std::max(partLocations[i].first.x, 0);
            partLocations[i].first.y = std::max(partLocations[i].first.y, 0);

            partLocations[i].first.x = std::min(partLocations[i].first.x,
                imgSize.width - partSize.width);
            partLocations[i].first.y = std::min(partLocations[i].first.y,
                imgSize.height - partSize.height);

            theta += 2.0 * M_PI / (nParts + 1.0);
        }
        DRWN_ASSERT(theta <= 2.0 * M_PI + DRWN_EPSILON);
    }

    for (int i = 0; i < nParts; i++) {
        DRWN_LOG_VERBOSE("...initializing part " << i << " of size "
            << toString(partLocations[i].second) << " at " << toString(partLocations[i].first));
    }

    return partLocations;
}

cv::Scalar drwnPartsModel::partColorMap(int v)
{
    return CV_RGB(((v + 1) & 0x04) != 0 ? 255 : 127,
        ((v + 1) & 0x02) != 0 ? 255 : 127, ((v + 1) & 0x01) != 0 ? 255 : 127);
}


// drwnTemplatePartsModel ----------------------------------------------------

drwnTemplatePartsModel::drwnTemplatePartsModel() : drwnPartsModel()
{
    // do nothing
}

drwnTemplatePartsModel::drwnTemplatePartsModel(const drwnTemplatePartsModel& model) :
    drwnPartsModel(model)
{
    // do nothing
}

drwnTemplatePartsModel::~drwnTemplatePartsModel()
{
    // do nothing
}

// learning
void drwnTemplatePartsModel::learnModel(int nParts, const vector<cv::Mat>& imgs)
{
    DRWN_ASSERT((nParts > 0) && (imgs.size() > 0));
    clear();

    // extract base size
    const int imgWidth = imgs.front().cols;
    const int imgHeight = imgs.front().rows;
    _baseSize = cv::Size(imgWidth, imgHeight);

    // compute features
    vector<cv::Mat> features;
    for (unsigned i = 0; i < imgs.size(); i++) {
        cv::Mat edges = drwnSoftEdgeMap(imgs[i], true);
        features.push_back(cv::Mat(edges.rows, edges.cols, CV_32FC1));
        edges.convertTo(features.back(), CV_32FC1);
    }

    // initialize part locations
    vector<pair<cv::Point, cv::Size> > partLocations = initializePartLocations(nParts,
        cv::Size(imgWidth, imgHeight));

    // initialize each part
    for (int i = 0; i < nParts; i++) {
        const int partHeight = partLocations[i].second.height;
        const int partWidth = partLocations[i].second.width;

        cv::Mat tmpl = cv::Mat::zeros(partHeight, partWidth, CV_32FC1);
        cv::Rect roi = cvRect(partLocations[i].first.x, partLocations[i].first.y, partWidth, partHeight);
        for (int j = 0; j < (int)features.size(); j++) {
            DRWN_ASSERT((features[j].rows == imgHeight) && (features[j].cols == imgWidth));
            cv::add(tmpl, features[j](roi), tmpl);
        }
        tmpl *= 1.0 / (double)features.size();

        _parts.push_back(new drwnPart(cv::Size(partWidth, partHeight), 1));
        _parts.back()->setWeights(tmpl);
        _parts.back()->setOffset(cv::Point(partLocations[i].first.x - imgWidth / 2,
                partLocations[i].first.y - imgHeight / 2));
        _parts.back()->setDCosts(drwnDeformationCost(0.0, 0.0, 1.0, 1.0));
    }
}

// computation of dense part matching costs
vector<cv::Mat> drwnTemplatePartsModel::computeMatchingCosts(const cv::Mat& img) const
{
    cv::Mat edgeMap = drwnSoftEdgeMap(img, true);
    cv::Mat features(img.rows, img.cols, CV_32FC1);
    edgeMap.convertTo(features, CV_32FC1);

    vector<cv::Mat> mcosts;
    for (unsigned i = 0; i < _parts.size(); i++) {
        mcosts.push_back(_parts[i]->unaryCosts(features));
    }

    return mcosts;
}

// drwnHOGPartsModel --------------------------------------------------------

int drwnHOGPartsModel::X_STEP_SIZE = 4;
int drwnHOGPartsModel::Y_STEP_SIZE = 4;

drwnHOGPartsModel::drwnHOGPartsModel() : drwnPartsModel()
{
    // do nothing
}

drwnHOGPartsModel::drwnHOGPartsModel(const drwnHOGPartsModel& model) :
    drwnPartsModel(model)
{
    // do nothing
}

drwnHOGPartsModel::~drwnHOGPartsModel()
{
    // do nothing
}

// learning
void drwnHOGPartsModel::learnModel(int nParts, const vector<cv::Mat>& imgs)
{
    DRWN_ASSERT((nParts > 0) && (imgs.size() > 0));
    clear();

    // extract base size
    const int imgWidth = imgs.front().cols;
    const int imgHeight = imgs.front().rows;
    _baseSize = cv::Size(imgWidth, imgHeight);

    // compute feature averages
    drwnHOGFeatures hog;
    vector<cv::Mat> weights;
    vector<cv::Mat> features;
    for (unsigned i = 0; i < imgs.size(); i++) {
        hog.computeFeatures(imgs[i], features);
        if (weights.empty()) {
            std::swap(weights, features);
        } else {
            for (unsigned j = 0; j < features.size(); j++) {
                weights[j] += features[j];
            }
        }
    }

    DRWN_ASSERT(!weights.empty());
    for (unsigned j = 0; j < weights.size(); j++) {
        weights[j] *= 1.0 / (double)imgs.size();
    }

    // initialize part locations
    const int featureWidth = weights.front().cols;
    const int featureHeight = weights.front().rows;

    vector<pair<cv::Point, cv::Size> > partLocations = initializePartLocations(nParts,
        cv::Size(featureWidth, featureHeight));

    // TODO: clean up mapping from pixel to feature coordinates

    // initialize each part
    for (int i = 0; i < nParts; i++) {
        const int featurePartWidth = partLocations[i].second.width;
        const int featurePartHeight = partLocations[i].second.height;
        const int partWidth = imgWidth * featurePartWidth / featureWidth;
        const int partHeight = imgHeight * featurePartHeight / featureHeight;

        cv::Rect roi(partLocations[i].first.x, partLocations[i].first.y,
            featurePartWidth, featurePartHeight);

        vector<cv::Mat> w(weights.size());
        for (int j = 0; j < (int)weights.size(); j++) {
            DRWN_LOG_DEBUG("...copying " << toString(roi) << " from " << toString(weights[j]));
            //weights[j](roi).copyTo(w[j]);

            w[j] = cv::Mat(roi.height, roi.width, CV_8UC1);
            weights[j](roi).copyTo(w[j]);
        }

        _parts.push_back(new drwnPart(cv::Size(partWidth, partHeight), (int)weights.size()));
        _parts.back()->setWeights(w);
        _parts.back()->setOffset(cv::Point(imgWidth * partLocations[i].first.x / featureWidth - imgWidth / 2,
                imgHeight * partLocations[i].first.y / featureHeight - imgHeight / 2));
        _parts.back()->setDCosts(drwnDeformationCost(0.0, 0.0, 1.0, 1.0));
    }
}

void drwnHOGPartsModel::learnModel(const vector<cv::Size>& partSizes,
    const vector<pair<cv::Mat, drwnPartsAssignment> >& imgs)
{
    DRWN_ASSERT((partSizes.size() > 0) && (imgs.size() > 0));

    drwnHOGPartsModel oldModel(*this);
    clear();

    drwnHOGFeatures hog;

    // initialize each part
    for (unsigned i = 0; i < partSizes.size(); i++) {
        cv::Size featureSize = hog.numBlocks(partSizes[i]);
        cv::Size cropSize = hog.padImageSize(partSizes[i]);
        vector<cv::Mat> w(hog.numFeatures());

        for (unsigned j = 0; j < w.size(); j++) {
            w[j] = cv::Mat::zeros(featureSize.height, featureSize.width, CV_32FC1);
        }

        drwnDeformationCost offsetStats(0.0, 0.0, 0.0, 0.0);
        int sampleCount = 0;
        for (unsigned j = 0; j < imgs.size(); j++) {
            // skip occluded parts
            if (imgs[j].second.occluded[i]) continue;

            // crop part
            // TODO: handle boundary issues
            cv::Rect roi(imgs[j].second.locations[i].x, imgs[j].second.locations[i].y,
                cropSize.width, cropSize.height);
            cv::Mat croppedPart = cv::Mat::zeros(roi.height, roi.width, imgs[j].first.type());
            drwnTruncateRect(roi, imgs[j].first.cols, imgs[j].first.rows);
            if (roi.width * roi.height == 0) {
                DRWN_LOG_ERROR(toString(roi) << " from " << toString(cv::Rect(imgs[j].second.locations[i].x,
                            imgs[j].second.locations[i].y, cropSize.width, cropSize.height)));
                continue;
            }

            imgs[j].first(roi).copyTo(croppedPart(cv::Rect(0, 0, roi.width, roi.height)));

            // featurize
            vector<cv::Mat> features;
            hog.computeFeatures(croppedPart, features);

            // accumulate
            for (unsigned k = 0; k < features.size(); k++) {
                DRWN_ASSERT_MSG((features[k].cols == w[k].cols) && (features[k].rows == w[k].rows),
                    i << ", " << j << " : " << features[k].cols << "-by-" << features[k].rows << " != "
                    << w[k].cols << "-by-" << w[k].rows << " : " << toString(roi) << " : " << toString(partSizes[i]));
                w[k] += features[k];
            }

            // update offset statistics
            offsetStats.dx += imgs[j].second.locations[i].x - imgs[j].first.cols / 2;
            offsetStats.dy += imgs[j].second.locations[i].y - imgs[j].first.rows / 2;
            offsetStats.dx2 += (imgs[j].second.locations[i].x - imgs[j].first.cols / 2) *
                (imgs[j].second.locations[i].x - imgs[j].first.cols / 2);
            offsetStats.dy2 += (imgs[j].second.locations[i].y - imgs[j].first.rows / 2) *
                (imgs[j].second.locations[i].y - imgs[j].first.rows / 2);

            sampleCount += 1;
        }

        for (unsigned j = 0; j < w.size(); j++) {
            w[j] /= (double)sampleCount;
        }
        offsetStats.dx /= (double)sampleCount;
        offsetStats.dy /= (double)sampleCount;
        offsetStats.dx2 = std::max(0.01, offsetStats.dx2 / (double)sampleCount -
            offsetStats.dx * offsetStats.dx);
        offsetStats.dy2 = std::max(0.01, offsetStats.dy2 / (double)sampleCount -
            offsetStats.dy * offsetStats.dy);

        _parts.push_back(new drwnPart(partSizes[i], (int)w.size()));
        _parts.back()->setWeights(w);
        _parts.back()->setOffset(cvPoint((int)offsetStats.dx, (int)offsetStats.dy));
        _parts.back()->setDCosts(drwnDeformationCost(0.0, 0.0, 1.0 / offsetStats.dx2, 1.0 / offsetStats.dy2));
    }
}

// feature computation
// computation of dense part matching costs
vector<cv::Mat> drwnHOGPartsModel::computeMatchingCosts(const cv::Mat& img) const
{
    DRWN_FCN_TIC;
    drwnHOGFeatures hog;
    pair<cv::Mat, cv::Mat> gradMagAndOri = hog.gradientMagnitudeAndOrientation(img);

    vector<cv::Mat> mcosts(_parts.size());
    for (unsigned i = 0; i < _parts.size(); i++) {
        mcosts[i] = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    }

    // increment locations by stepSize for speedup
    for (int y = 0; y < drwnHOGFeatures::DEFAULT_CELL_SIZE; y += Y_STEP_SIZE) {
        for (int x = 0; x < drwnHOGFeatures::DEFAULT_CELL_SIZE; x += X_STEP_SIZE) {
            // shift gradients
            cv::Rect roi(x, y, img.cols - x, img.rows - y);
            cv::Mat gm = gradMagAndOri.first(roi);
            cv::Mat go = gradMagAndOri.second(roi);

            // compute features
            vector<cv::Mat> features;
            hog.computeFeatures(make_pair(gm, go), features);

            // compute part costs
            for (unsigned i = 0; i < _parts.size(); i++) {
                cv::Mat cost = _parts[i]->unaryCosts(features);
                for (int yy = std::max(0, y - Y_STEP_SIZE / 2); yy <= std::min(y + Y_STEP_SIZE / 2 + 1, img.rows); yy++) {
                    for (int xx = std::max(0, x - X_STEP_SIZE / 2); xx <= std::min(x + X_STEP_SIZE / 2 + 1, img.cols); xx++) {
                        for (int v = 0; v < cost.rows; v++) {
                            if (v * drwnHOGFeatures::DEFAULT_CELL_SIZE + yy >= img.rows)
                                break;
                            const float *p = cost.ptr<const float>(v);
                            float *q = mcosts[i].ptr<float>(v * drwnHOGFeatures::DEFAULT_CELL_SIZE + yy);
                            for (int u = 0; u < cost.cols; u++) {
                                if (u * drwnHOGFeatures::DEFAULT_CELL_SIZE + xx >= img.cols)
                                    break;
                                q[u * drwnHOGFeatures::DEFAULT_CELL_SIZE + xx] = p[u];
                            }
                        }
                    }
                }
            }
        }
    }

    DRWN_FCN_TOC;
    return mcosts;
}

// drwnPartsModelConfig -----------------------------------------------------

class drwnPartsModelConfig : public drwnConfigurableModule {
public:
    drwnPartsModelConfig() : drwnConfigurableModule("drwnPartsModel") { }
    ~drwnPartsModelConfig() { }

    void usage(ostream &os) const {
        os << "      matchMode    :: part match mode (default: " << drwnPart::MATCH_MODE << ")\n";
        os << "      msgScale     :: message passing scale (default: " << drwnPartsInference::MESSAGE_PASSING_SCALE << ")\n";
        os << "      lambda       :: pairwise cost (default: " << drwnPartsInference::DEFAULT_LAMBDA << ")\n";
        os << "      occlusion    :: occlusion penalty (default: " << drwnPartsInference::DEFAULT_OCCLUSION_COST << ")\n";
        os << "      hogStepSize  :: step size for HOG model inference (default: "
           << drwnHOGPartsModel::X_STEP_SIZE << " and " << drwnHOGPartsModel::Y_STEP_SIZE << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "matchMode")) {
            drwnPart::MATCH_MODE = atoi(value);
        } else if (!strcmp(name, "msgScale")) {
            drwnPartsInference::MESSAGE_PASSING_SCALE = atoi(value);
        } else if (!strcmp(name, "lambda")) {
            drwnPartsInference::DEFAULT_LAMBDA = atof(value);
        } else if (!strcmp(name, "occlusion")) {
            drwnPartsInference::DEFAULT_OCCLUSION_COST = atof(value);
        } else if (!strcmp(name, "hogStepSize")) {
            drwnHOGPartsModel::X_STEP_SIZE = drwnHOGPartsModel::Y_STEP_SIZE = std::max(1, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnPartsModelConfig gPartsModelConfig;

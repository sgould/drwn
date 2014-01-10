/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVisionUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <vector>
#include <list>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"

#include "drwnMultiSegConfig.h"
#include "drwnOpenCVUtils.h"
#include "drwnPixelNeighbourContrasts.h"
#include "drwnSuperpixelContainer.h"
#include "drwnVisionUtils.h"

using namespace std;

cv::Rect drwnTransformROI(const cv::Rect& roi, const cv::Size& srcSize, const cv::Size& dstSize)
{
    cv::Rect r;
    r.x = roi.x * dstSize.width / srcSize.width;
    r.y = roi.y * dstSize.height / srcSize.height;
    r.width = (roi.x + roi.width) * dstSize.width / srcSize.width - r.x;
    r.height = (roi.y + roi.height) * dstSize.height / srcSize.height - r.y;
    return r;
}

// load an over-segmentation or pixel labeling
void drwnLoadPixelLabels(cv::Mat& pixelLabels, const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    string ext = strExtension(string(filename));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext.compare("png") == 0) {
        // load 8- or 16-bit png
        cv::Mat tmp = cv::imread(string(filename), CV_LOAD_IMAGE_ANYDEPTH);
        DRWN_ASSERT_MSG(tmp.data != NULL, filename);
        DRWN_LOG_DEBUG("drwnLoadPixelLabels read " << toString(tmp));

        //! \bug 8-bit png files with colour tables are automatically
        //! converted to greyscale images. We'd like to just load their
        //! indexes.
        if (tmp.depth() != IPL_DEPTH_16U) {
            DRWN_LOG_WARNING_ONCE("Darwin does not currently support 8-bit PNG files; using drwnMultiSegConfig.regionDefinitions to convert.");
            const MatrixXi labels = gMultiSegRegionDefs.convertImageToLabels(filename);

            if (pixelLabels.data == NULL) {
                pixelLabels = cv::Mat(labels.rows(), labels.cols(), CV_32SC1);
            }
            DRWN_ASSERT_MSG((labels.rows() == pixelLabels.rows) &&
                (labels.cols() == pixelLabels.cols), "size mismatch");

            for (int y = 0; y < labels.rows(); y++) {
                for (int x = 0; x < labels.cols(); x++) {
                    pixelLabels.at<int>(y, x) = labels(y, x);
                }
            }
            return;
        }

        if (pixelLabels.data == NULL) {
            pixelLabels = cv::Mat(tmp.rows, tmp.cols, CV_32SC1);
        }
        DRWN_ASSERT_MSG((tmp.rows == pixelLabels.rows) &&
            (tmp.cols == pixelLabels.cols), "size mismatch");
        tmp.convertTo(pixelLabels, CV_32S);

    } else if (ext.compare("txt") == 0) {
        // integer text file
        ifstream ifs(filename);
        DRWN_ASSERT_MSG(!ifs.fail(), filename);
        int w = drwnCountFields(&ifs);
        DRWN_LOG_DEBUG(filename << " has width " << w);
        list<int> values;
        while (1) {
            int v;
            ifs >> v;
            if (ifs.fail()) break;
            values.push_back(v);
        }
        int h = values.size() / w;
        DRWN_LOG_DEBUG(filename << " has height " << h);
        if (pixelLabels.data == NULL) {
            pixelLabels = cv::Mat(h, w, CV_32SC1);
        }
        DRWN_ASSERT_MSG((pixelLabels.rows = h) &&
            (pixelLabels.cols = w), "size mismatch");
        int *p = pixelLabels.ptr<int>(0);
        list<int>::const_iterator it = values.begin();
        while (it != values.end()) {
            *p++ = *it++;
        }
    } else {
        DRWN_LOG_FATAL("unknown file extension ." << ext);
    }
}

void drwnLoadPixelLabels(cv::Mat& pixelLabels, const char *filename, int numLabels)
{
    drwnLoadPixelLabels(pixelLabels, filename);

    cv::Mat indx(pixelLabels.rows, pixelLabels.cols, CV_8UC1);
    cv::compare(pixelLabels, cv::Scalar::all(numLabels), indx, CV_CMP_GE);
    pixelLabels.setTo(cv::Scalar::all(-1), indx);
}

void drwnLoadPixelLabels(MatrixXi& pixelLabels, const char *filename, int numLabels)
{
    cv::Mat m;
    if (pixelLabels.size() != 0) {
        m = cv::Mat(pixelLabels.rows(), pixelLabels.cols(), CV_32SC1);
    }

    drwnLoadPixelLabels(m, filename);

    pixelLabels.resize(m.rows, m.cols);
    for (int y = 0; y < m.rows; y++) {
        const int *p = m.ptr<const int>(y);
        for (int x = 0; x < m.cols; x++) {
            pixelLabels(y, x) = (p[x] < numLabels ? p[x] : -1);
        }
    }
}

int drwnConnectedComponents(cv::Mat& segments, bool b8Connected)
{
    // make sure input is an integer type
    if ((segments.depth() == CV_8U) || (segments.depth() == CV_8S) || (segments.depth() == CV_16S)) {
        cv::Mat m(segments.rows, segments.cols, CV_32SC1);
        segments.convertTo(m, CV_32S);
        const int n = drwnConnectedComponents(m, b8Connected);
        m.convertTo(segments, segments.depth());
        return n;
    }

    if (segments.depth() != CV_32S) {
        DRWN_LOG_FATAL("drwnConnectedComponents only valid for integer matrices");
    }

    // initialize nodes in disjoint set by 4-connected raster scanning image
    const int width = segments.cols;
    const int height = segments.rows;

    cv::Mat cc(height, width, CV_32SC1);
    int *p = cc.ptr<int>(0);
    const int *q = segments.ptr<const int>(0);
    int totalNodes = 0;

    // first row
    *p++ = totalNodes++;
    q += 1;
    for (int x = 1; x < width; x++, p++, q++) {
        if (*q == *(q - 1)) {
            *p = *(p - 1);
        } else {
            *p = totalNodes++;
        }
    }

    // special case
    if (height == 1) {
        segments = cc;
        return totalNodes;
    }

    // remaining rows
    for (int y = 1; y < height; y++) {
        if (*q == *(q - width)) {
            *p = *(p - width);
        } else {
            *p = totalNodes++;
        }
        p++; q++;
        for (int x = 1; x < width; x++, p++, q++) {
            if (*q == *(q - 1)) {
                *p = *(p - 1);
            } else if (*q == *(q - width)) {
                *p = *(p - width);
            } else {
                *p = totalNodes++;
            }
        }
    }

    // initialize disjoint sets
    drwnDisjointSets components(totalNodes);

    // merge adjacent components with same id
    if (b8Connected) {
        p = cc.ptr<int>(0) + 1;
        q = segments.ptr<const int>(0) + 1;

        // 8-connected
        for (int x = 1; x < width; x++, q++, p++) {
            if ((*q == *(q + width - 1)) && (*p != *(p + width - 1))) {
                components.join(components.find(*p), components.find(*(p + width - 1)));
            }
        }

        for (int y = 1; y < height; y++) {
            // 4-connected
            if ((*q == *(q - width)) && (*p != *(p - width))) {
                components.join(components.find(*p), components.find(*(p- width)));
            }

            q += 1; p += 1;
            for (int x = 1; x < width; x++, q++, p++) {
                // 4-connected
                if ((*q == *(q - width)) && (*p != *(p - width))) {
                    components.join(components.find(*p), components.find(*(p - width)));
                }
                // 8-connected
                if ((*q == *(q - width - 1)) && (*p != *(p - width - 1))) {
                    components.join(components.find(*p), components.find(*(p - width - 1)));
                }
                if ((y < height - 1) && (*q == *(q + width - 1)) && (*p != *(p + width - 1))) {
                    components.join(components.find(*p), components.find(*(p + width - 1)));
                }
            }
        }

    } else {
        // 4-connected
        p = cc.ptr<int>(1);
        q = segments.ptr<const int>(1);
        for (int y = 1; y < height; y++) {
            if ((*q == *(q - width)) && (*p != *(p - width))) {
                components.join(components.find(*p), components.find(*(p - width)));
            }

            q += 1; p += 1;
            for (int x = 1; x < width; x++, q++, p++) {
                if ((*q == *(q - width)) && (*p != *(p - width))) {
                    components.join(components.find(*p), components.find(*(p - width)));
                }
            }
        }
    }

    // retrieve components
    int numComponents = 0;
    vector<int> componentIds(totalNodes, -1);
    int *qq = segments.ptr<int>(0);
    const int *pp = cc.ptr<const int>(0);
    for (int i = 0; i < width * height; i++) {
        if (componentIds[pp[i]] == -1) {
            int setId = components.find(pp[i]);
            if (componentIds[setId] == -1) {
                componentIds[setId] = numComponents++;
            }
            componentIds[pp[i]] = componentIds[setId];
        }
        qq[i] = componentIds[pp[i]];
    }

    return numComponents;
}

int drwnConnectedComponents(MatrixXi& segments, bool b8Connected)
{
    if (segments.size() == 0) return 0;

    cv::Mat m(segments.rows(), segments.cols(), CV_32SC1);
    for (int y = 0; y < m.rows; y++) {
        int *p = m.ptr<int>(y);
        for (int x = 0; x < m.cols; x++) {
            p[x] = segments(y, x);
        }
    }

    const int numComponents = drwnConnectedComponents(m, b8Connected);

    for (int y = 0; y < m.rows; y++) {
        const int *p = m.ptr<const int>(y);
        for (int x = 0; x < m.cols; x++) {
            segments(y, x) = p[x];
        }
    }

    return numComponents;
}

cv::Mat drwnFastSuperpixels(const cv::Mat& img, unsigned gridSize)
{
    DRWN_ASSERT((img.cols > 2) && (img.rows > 2) && (gridSize > 1));
    DRWN_FCN_TIC;
    DRWN_LOG_DEBUG("computing superpixels for " << toString(img) << "...");

    const unsigned hGrid = std::max(unsigned(2), (img.cols + gridSize - 2) / (gridSize - 1));
    const unsigned vGrid = std::max(unsigned(2), (img.rows + gridSize - 2) / (gridSize - 1));

    // compute pairwise contrasts
    drwnPixelNeighbourContrasts contrasts(img);

    // initialize maxflow graph
    const int nVariables = img.cols * img.rows;
    drwnBKMaxFlow g(nVariables);
    g.addNodes(nVariables);

    // unary terms
    for (int y = 0; y < img.rows; y++) {
        g.addTargetEdge(y * img.cols, DRWN_DBL_MAX);
        for (int x = hGrid; x < img.cols; x += hGrid) {
            g.addSourceEdge(y * img.cols + x - 1, DRWN_DBL_MAX);
            g.addTargetEdge(y * img.cols + x, DRWN_DBL_MAX);
        }
        g.addSourceEdge(y * img.cols + img.cols - 1, DRWN_DBL_MAX);
    }

    // vertical pairwise terms
    for (int y = 1; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            const double w = contrasts.contrastN(x, y);
            g.addEdge((y - 1) * img.cols + x, y * img.cols + x, w, w);
        }
    }

    // horizontal pairwise terms
    for (int y = 0; y < img.rows; y++) {
        for (int x = 1; x < img.cols; x++) {
            const double w = contrasts.contrastW(x, y);
            g.addEdge(y * img.cols + x - 1, y * img.cols + x, w, w);
        }
    }

    g.solve();

    // decode
    cv::Mat seg(img.rows, img.cols, CV_32SC1);
    for (int y = 0; y < seg.rows; y++) {
        for (int x = 0; x < seg.cols; x++) {
            seg.at<int32_t>(y, x) = int(x / hGrid) + (g.inSetS(y * img.cols + x) ? 1 : 0);
        }
    }

    // reset maxflow graph
    g.reset();

    // unary terms
    for (int x = 0; x < img.cols; x++) {
        g.addTargetEdge(x, DRWN_DBL_MAX);
    }
    for (int y = vGrid; y < img.rows; y += vGrid) {
        for (int x = 0; x < img.cols; x++) {
            g.addSourceEdge((y - 1) * img.cols + x, DRWN_DBL_MAX);
            g.addTargetEdge(y * img.cols + x, DRWN_DBL_MAX);
        }
    }
    for (int x = 0; x < img.cols; x++) {
        g.addSourceEdge((img.rows - 1) * img.cols + x, DRWN_DBL_MAX);
    }

    // vertical pairwise terms
    for (int y = 1; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            const double w = contrasts.contrastN(x, y);
            g.addEdge((y - 1) * img.cols + x, y * img.cols + x, w, w);
        }
    }

    // horizontal pairwise terms
    for (int y = 0; y < img.rows; y++) {
        for (int x = 1; x < img.cols; x++) {
            const double w = contrasts.contrastW(x, y);
            g.addEdge(y * img.cols + x - 1, y * img.cols + x, w, w);
        }
    }

    g.solve();

    // decode and renumber
    map<int, int> renumbering;
    for (int y = 0; y < seg.rows; y++) {
        for (int x = 0; x < seg.cols; x++) {
            const int n = img.cols * (int(y / vGrid) + (g.inSetS(y * img.cols + x) ? 1 : 0)) + seg.at<int32_t>(y, x);
            const map<int, int>::const_iterator it = renumbering.find(n);
            if (it != renumbering.end()) {
                seg.at<int32_t>(y, x) = it->second;
            } else {
                seg.at<int32_t>(y, x) = (int)renumbering.size();
                renumbering.insert(make_pair(n, (int)renumbering.size()));
            }
        }
    }

    DRWN_LOG_DEBUG("...generated " << ((int)renumbering.size() + 1) << " superpixels");
    DRWN_FCN_TOC;
    return seg;
}

cv::Mat drwnKMeansSegments(const cv::Mat& img, unsigned numCentroids)
{
    DRWN_ASSERT((img.depth() == CV_8U) && (img.channels() == 3) && (img.rows * img.cols > numCentroids));

    // extract data
    vector<vector<double> > data;
    data.reserve(img.rows * img.cols);

    vector<double> colour(3);
    for (int y = 0; y < img.rows; y++) {
        const unsigned char *p = img.ptr<const unsigned char>(y);
        for (int x = 0; x < img.cols; x++) {
            colour[2] = (double)p[3 * x + 0] / 255.0;
            colour[1] = (double)p[3 * x + 1] / 255.0;
            colour[0] = (double)p[3 * x + 2] / 255.0;
            data.push_back(colour);
        }
    }

    // whiten data
    drwnPCA pca;
    pca.train(data);
    pca.transform(data);

    // learn model
    drwnKMeans model(numCentroids);
    model.train(data);

    // extract segmentation
    vector<double> d;
    cv::Mat seg(img.rows, img.cols, CV_32SC1, cv::Scalar(-1));
    int indx = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            model.transform(data[indx++], d);
            seg.at<int>(y, x) = drwn::argmin(d);
        }
    }

    return seg;
}

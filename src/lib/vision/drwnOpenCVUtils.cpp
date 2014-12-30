/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <vector>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnOpenCVUtils.h"

using namespace std;
using namespace Eigen;

// globals
unsigned drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT = DRWN_INT_MAX;
unsigned drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH = DRWN_INT_MAX;

// convert opencv data structures to a string
string toString(const cv::Mat& m)
{
    std::stringstream s;
    s << m.cols << "-by-" << m.rows
      << " (" << (8 * m.elemSize1()) << "-bit, "
      << m.channels() << "-channel)";
    return s.str();
}

string toString(const vector<cv::Mat>& vm)
{
    std::stringstream s;
    for (size_t i = 0; i < vm.size(); i++) {
        s << "cv::Mat " << i << ": "
          << vm[i].cols << "-by-" << vm[i].rows
	  << " (" << (8 * vm[i].elemSize1()) << "-bit, "
	  << vm[i].channels() << "-channel)\n";
    }

    return s.str();
}

string toString(const cv::Rect& r)
{
    std::stringstream s;
    s << "[" << r.x << ", " << r.y << ", " << r.width << ", " << r.height << "]";
    return s.str();
}


string toString(const cv::Point& pt)
{
    std::stringstream s;
    s << "(" << pt.x << ", " << pt.y << ")";
    return s.str();
}

string toString(const cv::Size& sz)
{
    std::stringstream s;
    s << sz.width << "-by-" << sz.height;
    return s.str();
}

string toString(const cv::Scalar& slr)
{
    std::stringstream s;
    s << "(" << slr[0] << ", " << slr[1] << ", " << slr[2] << ", " << slr[3] << ")";
    return s.str();
}


// operators
bool operator==(const CvRect& r, const CvRect& s)
{
    return ((r.x == s.x) && (r.y == s.y) && (r.width == s.width) && (r.height == s.height));
}

bool operator<(const CvSize& r, const CvSize& s)
{
    return ((r.height < s.height) || ((r.height == s.height) && (r.width < s.width)));
}

bool operator<(const CvPoint& p, const CvPoint& q)
{
    return ((p.y < q.y) || (p.x < q.x));
}

int drwnCmpCount(const cv::Mat& s, const cv::Mat& t, int cmpOp)
{
    cv::Mat m(s.rows, s.cols, CV_8UC1);
    cv::compare(s, t, m, cmpOp);
    return cv::countNonZero(m);
}

// show an image and wait
int drwnShowDebuggingImage(const cv::Mat& img, const std::string& name, bool bWait)
{
    DRWN_ASSERT(img.data != NULL);

    // convert if not 8-bit unsigned
    if (img.depth() != CV_8U) {
        double lb, ub;
        cv::minMaxLoc(img, &lb, &ub);
        if (ub == lb) { ub += 1.0; lb -= 1.0; }
        cv::Mat tmp(img.rows, img.cols, CV_8U);
        img.convertTo(tmp, CV_8U, 255.0 / (ub - lb), -lb * 255.0 / (ub - lb));
        return drwnShowDebuggingImage(tmp, name, bWait);
    }

    // rescale if too big
    if (((unsigned)img.rows > drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT) ||
        ((unsigned)img.cols > drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH)) {
        const double scale = std::min((double)drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT / (double)img.rows,
            (double)drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH / (double)img.cols);

        cv::Mat tmp((int)(scale * img.rows), (int)(scale * img.cols), img.type());
        cv::resize(img, tmp, tmp.size(), 0, 0, CV_INTER_LINEAR);
        return drwnShowDebuggingImage(tmp, name, bWait);
    }

    // show the image
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);

    int ch = -1;
    if (bWait) {
        ch = cv::waitKey(-1);
        if (ch == (int)'w') {
            const string filename = name + string(".png");
            cv::imwrite(filename, img);
        }
        cv::destroyWindow(name);
    } else {
        ch = cv::waitKey(30);
    }

    return ch;
}

int drwnShowDebuggingImage(const vector<cv::Mat>& views, const std::string& name, bool bWait)
{
    cv::Mat canvas = drwnCombineImages(views);
    return drwnShowDebuggingImage(canvas, name, bWait);
}

cv::Rect drwnFitBoundingBox(const vector<cv::Rect> &rects)
{
    if (rects.empty()) {
        return cv::Rect(0, 0, 0, 0);
    }

    cv::Rect r(rects[0]);
    for (unsigned i = 1; i < rects.size(); i++) {
        r |= rects[i];
    }

    return r;
}

cv::Rect drwnFitBoundingBox(const vector<cv::Point> &points)
{
    if (points.empty()) {
        return cv::Rect(0, 0, 0, 0);
    }

    cv::Rect r(points[0].x, points[0].y, 1, 1);
    for (unsigned i = 0; i < points.size(); i++) {
        r |= cv::Rect(points[i].x, points[i].y, 1, 1);
    }

    return r;
}

cv::Rect drwnFitBoundingBox(cv::Rect r, double aspectRatio)
{
    if (r.width > aspectRatio * r.height) {
        int delta = (int)(r.width / aspectRatio) - r.height;
        r.y -= delta / 2;
        r.height += delta;
    } else {
        int delta = (int)(r.height * aspectRatio) - r.width;
        r.x -= delta / 2;
        r.width += delta;
    }

    return r;
}

// area of intersection
double areaOverlap(const cv::Rect& r, const cv::Rect& s)
{
    int iw, ih;

    // find region of overlap
    iw = std::min(r.x + r.width, s.x + s.width) - std::max(r.x, s.x);
    ih = std::min(r.y + r.height, s.y + s.height) - std::max(r.y, s.y);

    if ((iw <= 0) || (ih <= 0)) {
        return 0.0;
    }

    // return area of intersection
    return (double)(iw * ih);

}

// convert image to greyscale (floating point)
cv::Mat drwnGreyImage(const cv::Mat& src)
{
    cv::Mat dst;
    if (src.channels() == 3) {
        cv::Mat tmp(src.rows, src.cols, src.depth());
        cv::cvtColor(src, tmp, CV_RGB2GRAY);
        if (tmp.depth() == CV_8U) {
            dst = cv::Mat(src.rows, src.cols, CV_32F);
            tmp.convertTo(dst, CV_32F, 1.0 / 255.0);
        } else {
            DRWN_ASSERT(tmp.depth() == CV_32F);
            dst = tmp;
        }
    } else {
        if (src.depth() == CV_8U) {
            dst = cv::Mat(src.rows, src.cols, CV_32F);
            src.convertTo(dst, CV_32F, 1.0 / 255.0);
        } else {
            DRWN_ASSERT(src.depth() == CV_32F);
            dst = src.clone();
        }
    }

    return dst;
}

cv::Mat drwnColorImage(const cv::Mat& src)
{
    DRWN_ASSERT(src.data != NULL);

    cv::Mat dst(src.rows, src.cols, CV_8UC3);
    if (src.channels() == 1) {
        if (src.depth() != CV_8U) {
            cv::Mat tmp(src.rows, src.cols, CV_8U);
            src.convertTo(tmp, CV_8U, 255.0);
            cv::cvtColor(tmp, dst, CV_GRAY2RGB);
        } else {
            cv::cvtColor(src, dst, CV_GRAY2RGB);
        }
    } else {
        if (src.depth() != CV_8U) {
            src.convertTo(dst, CV_8UC3, 255.0);
        } else {
            src.copyTo(dst);
        }
    }

    return dst;
}

cv::Mat drwnSoftEdgeMap(const cv::Mat& src, bool bNormalize)
{
    cv::Mat grey = drwnGreyImage(src);
    DRWN_ASSERT(grey.depth() == CV_32F);

    cv::Mat Dx(grey.rows, grey.cols, CV_32F);
    cv::Mat Dy(grey.rows, grey.cols, CV_32F);
    cv::Sobel(grey, Dx, CV_32F, 1, 0, 3);
    cv::Sobel(grey, Dy, CV_32F, 0, 1, 3);

    for (int y = 0; y < grey.rows; y++) {
        float *pm = grey.ptr<float>(y);
        const float *pDx = Dx.ptr<float>(y);
        const float *pDy = Dy.ptr<float>(y);
        for (int x = 0; x < grey.cols; x++) {
            pm[x] = sqrt((pDx[x] * pDx[x]) + (pDy[x] * pDy[x]));
        }
    }

    if (bNormalize) {
        drwnScaleToRange(grey, 0.0, 1.0);
    }

    return grey;
}

void drwnGreyImageInplace(cv::Mat& img)
{
    if ((img.channels() != 1) || (img.depth() != CV_32F)) {
        img = drwnGreyImage(img);
    }
}

void drwnColorImageInplace(cv::Mat& img)
{
    if ((img.channels() != 3) || (img.depth() != CV_8U)) {
        img = drwnColorImage(img);
    }
}

// pad image and copy boundary
cv::Mat drwnPadImage(const cv::Mat& src, int margin)
{
    return drwnPadImage(src, cv::Rect(-margin/2, -margin/2, src.cols + margin, src.rows + margin));
}

cv::Mat drwnPadImage(const cv::Mat& src, const cv::Rect& page)
{
    DRWN_ASSERT((src.data != NULL) && (page.x <= 0) && (page.y <= 0) &&
        (page.x + page.width >= src.cols) && (page.y + page.height >= src.rows));
    cv::Mat paddedImg(page.height, page.width, src.type());

    // copy image to (0,0) of page
    src.copyTo(paddedImg(cv::Rect(-page.x, -page.y, src.cols, src.rows)));

    // pad with mirroring
    if (page.x < 0) {
        cv::flip(src(cv::Rect(0, 0, -page.x, src.rows)),
            paddedImg(cv::Rect(0, -page.y, -page.x, src.rows)), 1);
    }

    if (page.x + page.width > src.cols) {
        const int M = page.width + page.x - src.cols;
        cv::flip(src(cv::Rect(src.cols - M, 0, M, src.rows)),
            paddedImg(cv::Rect(page.width - M, -page.y, M, src.rows)), 1);
    }

    if (page.y < 0) {
        cv::flip(src(cv::Rect(0, 0, src.cols, -page.y)),
            paddedImg(cv::Rect(-page.x, 0, src.cols, -page.y)), 0);
    }

    if (page.y + page.height > src.rows) {
        const int M = page.height + page.y - src.rows;
        cv::flip(src(cv::Rect(0, src.rows - M, src.cols, M)),
            paddedImg(cv::Rect(-page.x, page.height - M, src.cols, M)), 0);
    }

    if ((page.x < 0) && (page.y < 0)) {
        cv::flip(src(cv::Rect(0, 0, -page.x, -page.y)),
            paddedImg(cv::Rect(0, 0, -page.x, -page.y)), -1);
    }

    if ((page.x < 0) && (page.y + page.height > src.rows)) {
        const int M = page.height + page.y - src.rows;
        cv::flip(src(cv::Rect(0, src.rows - M, -page.x, M)),
            paddedImg(cv::Rect(0, page.height - M, -page.x, M)), -1);
    }

    if ((page.x + page.width > src.cols) && (page.y < 0)) {
        const int M = page.width + page.x - src.cols;
        cv::flip(src(cv::Rect(src.cols - M, 0, M, -page.y)),
            paddedImg(cv::Rect(page.width - M, 0, M, -page.y)), -1);
    }

    if ((page.x + page.width > src.cols) && (page.y + page.height > src.rows)) {
        const int M = page.width + page.x - src.cols;
        const int N = page.height + page.y - src.rows;
        cv::flip(src(cv::Rect(src.cols - M, src.rows - N, M, N)),
            paddedImg(cv::Rect(page.width - M, page.height - N, M, N)), -1);
    }

    return paddedImg;
}

// translate an array
cv::Mat drwnTranslateMatrix(const cv::Mat& matrix, const cv::Point& origin, double fillValue)
{
    cv::Mat translated(matrix.rows, matrix.cols, matrix.type(), cv::Scalar::all(fillValue));

    const cv::Rect srcRect(std::max(0, origin.x), std::max(0, origin.y),
        matrix.cols - abs(origin.x), matrix.rows - abs(origin.y));
    const cv::Rect dstRect(std::max(0, -origin.x), std::max(0, -origin.y),
        matrix.cols - abs(origin.x), matrix.rows - abs(origin.y));

    translated(dstRect) = matrix(srcRect);

    return translated;
}

// rotate an image
cv::Mat drwnRotateImage(const cv::Mat& img, float theta)
{
    DRWN_ASSERT(img.data != NULL);
    cv::Point2f origin(0.5f * img.cols, 0.5f * img.rows);

    cv::Mat R = cv::getRotationMatrix2D(origin, 180.0f * theta / M_PI, 1.0);

    cv::Mat dst(img.rows, img.cols, img.type());
    cv::warpAffine(img, dst, R, dst.size());

    return dst;
}

// scale image or matrix
void drwnScaleToRange(cv::Mat &m, double minValue, double maxValue)
{
    DRWN_ASSERT(minValue <= maxValue);
    double l, u;
    cv::minMaxLoc(m, &l, &u);
    if (l != u) {
        m *= (maxValue - minValue) / (u - l);
        m += (minValue - l * (maxValue - minValue) / (u - l));
    } else {
        m.setTo(0.5 * (maxValue + minValue));
    }

}

// resize image inplace
void drwnResizeInPlace(cv::Mat& m, const cv::Size& size, int interpolation)
{
    drwnResizeInPlace(m, size.height, size.width, interpolation);
}

// resize matrix inplace
void drwnResizeInPlace(cv::Mat &m, int rows, int cols, int interpolation)
{
    if ((m.rows == rows) && (m.cols == cols))
        return;

    cv::Mat tmp(rows, cols, m.type());
    cv::resize(m, tmp, tmp.size(), 0, 0, interpolation);
    m = tmp;
}

// crop an image or matrix inplace
void drwnCropInPlace(cv::Mat& image, cv::Rect roi)
{
    drwnTruncateRect(roi, image);
    if ((roi.x == 0) && (roi.y == 0) && (roi.width == image.cols) && (roi.height == image.rows)) {
        return;
    }

    const cv::Mat tmp = image(roi).clone();
    image = tmp;
}

bool drwnValidRect(const cv::Rect& r, int width, int height)
{
    return ((r.x >= 0) && (r.y >= 0) && (r.width > 0) && (r.height > 0) &&
        (r.x + r.width <= width) && (r.y + r.height <= height));
}

// truncates a rectangle to fit inside the image
void drwnTruncateRect(cv::Rect& r, int width, int height)
{
    if (r.x < 0) {
        r.width += r.x; // decrease the width by that amount
        r.x = 0;
    }
    if (r.y < 0) {
        r.height += r.y; // decrease the height by that amount
        r.y = 0;
    }
    if (r.x + r.width > width) r.width = width - r.x;
    if (r.y + r.height > height) r.height = height - r.y;
    if (r.width < 0) r.width = 0;
    if (r.height < 0) r.height = 0;
}

// combine images
cv::Mat drwnCombineImages(const vector<cv::Mat>& images, int rows, int cols)
{
    // check sizes
    DRWN_ASSERT(!images.empty());
    if ((rows <= 0) && (cols <= 0)) {
	cols = (int)ceil(sqrt((float)images.size()));
        rows = (int)ceil((double)images.size() / cols);
    } else if (rows <= 0) {
	rows = (int)ceil((float)images.size() / cols);
    } else if (cols <= 0) {
	cols = (int)ceil((float)images.size() / rows);
    } else {
	DRWN_ASSERT((int)images.size() <= rows * cols);
    }

    int maxWidth = 0;
    int maxHeight = 0;
    for (unsigned i = 0; i < images.size(); i++) {
        maxWidth = std::max(maxWidth, images[i].cols);
        maxHeight = std::max(maxHeight, images[i].rows);
    }
    DRWN_ASSERT((maxWidth > 0) && (maxHeight > 0));

    cv::Mat outImg = cv::Mat::zeros(maxHeight * rows, maxWidth * cols, images.front().type());

    for (unsigned i = 0; i < images.size(); i++) {
	if (images[i].data == NULL) continue;
	const int x = (i % cols) * maxWidth + (maxWidth - images[i].cols) / 2;
	const int y = ((int)(i / cols)) * maxHeight + (maxHeight - images[i].rows) / 2;
        images[i].copyTo(outImg(cv::Rect(x, y, images[i].cols, images[i].rows)));
    }

    return outImg;
}

//! \todo refactor to use drwnCreateHeatMap(m, vector<CvScalar>) variant
cv::Mat drwnCreateHeatMap(const cv::Mat& m, drwnColorMap cm)
{
    DRWN_ASSERT(m.data != NULL);

    // rescale matrix and call self if argument is not 32-bit floating point
    if (m.depth() != CV_32F) {
        cv::Mat tmp(m.rows, m.cols, CV_32F);
        m.convertTo(tmp, CV_32F, 1.0 / 255.0);
        drwnScaleToRange(tmp, 0.0, 1.0);
        cv::Mat heatmap = drwnCreateHeatMap(tmp, cm);
        return heatmap;
    }

    // all other matrices must be floating point
    DRWN_ASSERT((m.depth() == CV_32F) && (m.channels() == 1));
    cv::Mat heatMap(m.rows, m.cols, CV_8UC3);
    for (int y = 0; y < m.rows; y++) {
        const float *p = m.ptr<const float>(y);
        unsigned char *q = heatMap.ptr<unsigned char>(y);
        for (int x = 0; x < m.cols; x++, p++, q += 3) {
            unsigned char red = 0x00;
            unsigned char green = 0x00;
            unsigned char blue = 0x00;

            switch (cm) {
                case DRWN_COLORMAP_RAINBOW:
                {
                    int h = (int)(5 * (*p));
                    unsigned char color = (unsigned char)(255 * (5.0 * (*p) - (double)h));
                    switch (h) {
                    case 0:
                        red = 0x00; green = 0x00; blue = color; break;
                    case 1:
                        red = 0x00; green = color; blue = 0xff; break;
                    case 2:
                        red = 0x00; green = 0xff; blue = 0xff - color; break;
                    case 3:
                        red = color; green = 0xff; blue = 0x00; break;
                    case 4:
                        red = 0xff; green = 0xff - color; blue = 0x00; break;
                    default:
                        red = 0xff; green = 0x00; blue = 0x00; break;
                    }
                }
                break;

                case DRWN_COLORMAP_HOT:
                {
                    int h = (int)(2 * (*p));
                    unsigned char color = (unsigned char)(255 * (2.0 * (*p) - (double)h));
                    switch (h) {
                    case 0:
                        red = color; green = 0x00; blue = 0x00; break;
                    case 1:
                        red = 0xff; green = color; blue = 0x00; break;
                    default:
                        red = 0xff; green = 0xff; blue = 0x00; break;
                    }
                }
                break;

                case DRWN_COLORMAP_COOL:
                {
                    unsigned char color = (unsigned char)(255 * (*p));
                    red = color; green = 0xff - color; blue = 0xff;
                }
                break;

                case DRWN_COLORMAP_REDGREEN:
                {
                    int h = (int)(2 * (*p));
                    unsigned char color = (unsigned char)(255 * (2.0 * (*p) - (double)h));
                    switch (h) {
                    case 0:
                        red = 0x00; green = 0xff - color; blue = 0x00; break;
                    case 1:
                        red = color; green = 0x00; blue = 0x00; break;
                    default:
                        red = 0xff; green = 0x00; blue = 0x00; break;
                    }
                }
                break;

                case DRWN_COLORMAP_ANU:
                {
                    // red: 175, 30, 45
                    // blue: 148, 176, 188
#if 0
                    int h = (int)(2 * (*p));
                    double residual = (2.0 * (*p) - (double)h);
                    switch (h) {
                    case 0:
                        red = (unsigned char)(175 - 175 * residual);
                        green = (unsigned char)(30 - 30 * residual);
                        blue = (unsigned char)(45 - 45 * residual);
                        break;
                    case 1:
                        red = (unsigned char)(148 * residual);
                        green = (unsigned char)(176 * residual);
                        blue = (unsigned char)(188 * residual);
                        break;
                    default:
                        red = 175; green = 30; blue = 45; break;
                    }
#else
                    red = (unsigned char)(175 + (148 - 175) * (*p));
                    green = (unsigned char)(30 + (176 - 30) * (*p));
                    blue = (unsigned char)(45 + (188 - 45) * (*p));
#endif
                }
                break;

                default:
                    DRWN_LOG_FATAL("unknown colormap type");
            }

            q[0] = blue;
            q[1] = green;
            q[2] = red;
        }
    }

    return heatMap;
}

cv::Mat drwnCreateHeatMap(const cv::Mat& m, cv::Scalar colourA, cv::Scalar colourB)
{
    DRWN_ASSERT(m.data != NULL);

    // rescale matrix and call self if argument is not 32-bit floating point
    if (m.depth() != CV_32F) {
        cv::Mat tmp(m.rows, m.cols, CV_32F);
        m.convertTo(tmp, CV_32F, 1.0 / 255.0);
        drwnScaleToRange(tmp, 0.0, 1.0);
        cv::Mat heatmap = drwnCreateHeatMap(tmp, colourA, colourB);
        return heatmap;
    }

    // all other matrices must be floating point
    DRWN_ASSERT((m.depth() == CV_32F) && (m.channels() == 1));
    cv::Mat heatMap(m.rows, m.cols, CV_8UC3);
    for (int y = 0; y < m.rows; y++) {
        const float *p = m.ptr<const float>(y);
        unsigned char *q = heatMap.ptr<unsigned char>(y);
        for (int x = 0; x < m.cols; x++, p++, q += 3) {
            q[0] = (unsigned char)(colourA.val[0] + (colourB.val[0] - colourA.val[0]) * (*p));
            q[1] = (unsigned char)(colourA.val[1] + (colourB.val[1] - colourA.val[1]) * (*p));
            q[2] = (unsigned char)(colourA.val[2] + (colourB.val[2] - colourA.val[2]) * (*p));
        }
    }

    return heatMap;
}

cv::Mat drwnCreateHeatMap(const cv::Mat& m, const vector<cv::Scalar>& colours)
{
    DRWN_ASSERT((m.data != NULL) && (colours.size() > 1));

    // rescale matrix and call self if argument is not 32-bit floating point
    if (m.depth() != CV_32F) {
        cv::Mat tmp(m.rows, m.cols, CV_32F);
        m.convertTo(tmp, CV_32F, 1.0 / 255.0);
        drwnScaleToRange(tmp, 0.0, 1.0);
        cv::Mat heatmap = drwnCreateHeatMap(tmp, colours);
        return heatmap;
    }

    // all other matrices must be floating point
    const int nColours = (int)colours.size();
    DRWN_ASSERT((m.depth() == CV_32F) && (m.channels() == 1));
    cv::Mat heatMap(m.rows, m.cols, CV_8UC3);
    for (int y = 0; y < m.rows; y++) {
        const float *p = m.ptr<const float>(y);
        unsigned char *q = heatMap.ptr<unsigned char>(y);
        for (int x = 0; x < m.cols; x++, p++, q += 3) {
            const int h = (int)((nColours - 1) * (*p));
            const double residual = ((double)(nColours - 1) * (*p) - (double)h);
            cv::Scalar colourA = colours[h];
            cv::Scalar colourB = colours[std::min(h + 1, nColours - 1)];

            q[0] = (unsigned char)(colourA.val[0] + (colourB.val[0] - colourA.val[0]) * residual);
            q[1] = (unsigned char)(colourA.val[1] + (colourB.val[1] - colourA.val[1]) * residual);
            q[2] = (unsigned char)(colourA.val[2] + (colourB.val[2] - colourA.val[2]) * residual);
        }
    }

    return heatMap;
}

vector<cv::Scalar> drwnCreateColorTable(unsigned n, drwnColorMap cm)
{
    cv::Mat m(n, 1, CV_32F);
    for (unsigned i = 0; i < n; i++) {
        m.at<float>(i, 0) = (float)i / (float)n;
    }

    cv::Mat heatMap = drwnCreateHeatMap(m, cm);
    vector<cv::Scalar> colorTable(n, cv::Scalar(0));
    for (unsigned i = 0; i < n; i++) {
        colorTable[i] = CV_RGB(heatMap.at<unsigned char>(i, 2),
            heatMap.at<unsigned char>(i, 1), heatMap.at<unsigned char>(i, 0));
    }

    return colorTable;
}

void drwnDrawBoundingBox(cv::Mat& canvas, const cv::Rect& roi, cv::Scalar fgcolor,
    cv::Scalar bgcolor, int lineWidth)
{
    DRWN_ASSERT(canvas.data != NULL);
    if ((roi.width < 1) || (roi.height < 1))
        return;

    cv::rectangle(canvas, roi.tl(), roi.br(), bgcolor, lineWidth + 2, CV_AA);
    cv::rectangle(canvas, roi.tl(), roi.br(), fgcolor, lineWidth, CV_AA);
}

void drwnDrawPolygon(cv::Mat& canvas, const vector<cv::Point> &poly, cv::Scalar fgcolor,
    cv::Scalar bgcolor, int lineWidth, bool bClose)
{
    DRWN_ASSERT(canvas.data != NULL);
    if (poly.size() < 2) return;

    // background
    for (unsigned i = 0; i < poly.size() - 1; i++) {
        cv::line(canvas, poly[i], poly[i + 1], bgcolor, lineWidth + 2, CV_AA);
    }
    if (bClose) {
        cv::line(canvas, poly.back(), poly.front(), bgcolor, lineWidth + 2, CV_AA);
    }

    // foreground
    for (unsigned i = 0; i < poly.size() - 1; i++) {
        cv::line(canvas, poly[i], poly[i + 1], fgcolor, lineWidth, CV_AA);
    }
    if (bClose) {
        cv::line(canvas, poly.back(), poly.front(), fgcolor, lineWidth, CV_AA);
    }
}

void drwnDrawFullLinePlot(cv::Mat& canvas, const vector<double>& points,
    const cv::Scalar& lineColour, unsigned lineWidth,
    const cv::Scalar& baseShading, double baseAlpha)
{
    DRWN_ASSERT((canvas.data != NULL) && (canvas.channels() == 3));
    if (points.size() < 2) {
        DRWN_LOG_WARNING("not enough points in drwnDrawFullLinePlot");
        return;
    }

    const double sx = (double)canvas.cols / (double)points.size();
    const double sy = (double)canvas.rows;

    // create polygon
    cv::Point *p = new cv::Point[points.size() + 2];
    for (unsigned i = 0; i < points.size(); i++) {
        p[i] = cv::Point(i * sx, canvas.rows - points[i] * sy);
    }
    p[points.size()] = cv::Point(canvas.cols, canvas.rows);
    p[points.size() + 1] = cv::Point(0, canvas.rows);

    // shade below the graph
    baseAlpha = std::min(baseAlpha, 1.0);
    if (baseAlpha > 0.0) {
        cv::Mat mask = cv::Mat::zeros(canvas.rows, canvas.cols, CV_8UC1);
        int n = points.size() + 2;
        cv::fillPoly(mask, (const cv::Point **)&p, &n, 1, cv::Scalar::all(255));
        drwnShadeRegion(canvas, mask, baseShading, baseAlpha);
    }

    // draw the graph
    int n = points.size();
    cv::polylines(canvas, (const cv::Point **)&p, &n, 1, false, lineColour, lineWidth);

    // free polygon
    delete[] p;
}

void drwnDrawTarget(cv::Mat& canvas, const cv::Point& center,
    cv::Scalar color, int size, int lineWidth)
{
    DRWN_ASSERT(canvas.data != NULL);
    cv::circle(canvas, center, size, color, lineWidth);
    cv::line(canvas, cv::Point(center.x - (int)(1.5 * size), center.y),
        cv::Point(center.x + (int)(1.5 * size), center.y), color, lineWidth);
    cv::line(canvas, cv::Point(center.x, center.y - (int)(1.5 * size)),
        cv::Point(center.x, center.y + (int)(1.5 * size)), color, lineWidth);
}


// mouse event callback
void drwnOnMouse(int event, int x, int y, int flags, void *ptr)
{
    DRWN_ASSERT(ptr != NULL);
    drwnMouseState *mouseState = (drwnMouseState *)ptr;
    mouseState->event = event;
    mouseState->x = x;
    mouseState->y = y;
    mouseState->flags = flags;
}

vector<cv::Point> drwnWaitMouse(const string& windowName, const cv::Mat& img, int numPoints)
{
    DRWN_ASSERT(img.data != NULL);
    drwnMouseState mouseState;
    vector<cv::Point> mousePoints;

    bool bDestoryWindow = (cvGetWindowHandle(windowName.c_str()) == NULL);
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, drwnOnMouse, (void *)&mouseState);

    cv::Mat canvas(img.rows, img.cols, CV_8UC3);

    while ((int)mousePoints.size() < numPoints) {
        // redraw canvas
        img.convertTo(canvas, canvas.depth());
        for (int i = 0; i < (int)mousePoints.size(); i++) {
            drwnDrawTarget(canvas, mousePoints[i], CV_RGB(0, 255, 0));
        }

        // black target sight
        cv::circle(canvas, cv::Point(mouseState.x, mouseState.y), 5, CV_RGB(0, 0, 0), 2);
        cv::line(canvas, cv::Point(0, mouseState.y), cv::Point(canvas.cols, mouseState.y),
            CV_RGB(0, 0, 0), 2);
        cv::line(canvas, cv::Point(mouseState.x, 0), cv::Point(mouseState.x, canvas.rows),
            CV_RGB(0, 0, 0), 2);

        // red target sight
        cv::circle(canvas, cv::Point(mouseState.x, mouseState.y), 5, CV_RGB(255, 0, 0), 1);
        cv::line(canvas, cv::Point(0, mouseState.y), cv::Point(canvas.cols, mouseState.y),
            CV_RGB(255, 0, 0), 1);
        cv::line(canvas, cvPoint(mouseState.x, 0), cvPoint(mouseState.x, canvas.rows),
            CV_RGB(255, 0, 0), 1);

        cv::imshow(windowName, canvas);

        // wait for mouse click (or key press)
        while (mouseState.event == -1) {
            int ch = cv::waitKey(30);
            if (ch != -1) break;
        }

        // check event type
        if (mouseState.event == -1) {
            break;
        } else if (mouseState.event == CV_EVENT_LBUTTONUP) {
            mousePoints.push_back(cv::Point(mouseState.x, mouseState.y));
        }

        mouseState.event = -1;
    }

    cv::setMouseCallback(windowName, NULL);
    if (bDestoryWindow) {
        cv::destroyWindow(windowName);
    }

    return mousePoints;
}

cv::Rect drwnInputBoundingBox(const string& windowName, const cv::Mat& img)
{
    drwnMouseState mouseState;

    bool bDestoryWindow = (cvGetWindowHandle(windowName.c_str()) == NULL);
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, drwnOnMouse, (void *)&mouseState);

    cv::Mat canvas(img.rows, img.cols, CV_8UC3);

    cv::Rect boundingBox(0, 0, -1, -1);
    int clickCount = 0;
    while (clickCount < 2) {
        // redraw canvas
        img.convertTo(canvas, CV_8UC3);
        if (clickCount == 0) {
            // black target sight
            cv::circle(canvas, cv::Point(mouseState.x, mouseState.y), 5, CV_RGB(0, 0, 0), 2);
            cv::line(canvas, cv::Point(0, mouseState.y), cv::Point(canvas.cols, mouseState.y), CV_RGB(0, 0, 0), 2);
            cv::line(canvas, cv::Point(mouseState.x, 0), cv::Point(mouseState.x, canvas.rows), CV_RGB(0, 0, 0), 2);

            // red target sight
            cv::circle(canvas, cv::Point(mouseState.x, mouseState.y), 5, CV_RGB(255, 0, 0), 1);
            cv::line(canvas, cv::Point(0, mouseState.y), cv::Point(canvas.cols, mouseState.y), CV_RGB(255, 0, 0), 1);
            cv::line(canvas, cv::Point(mouseState.x, 0), cv::Point(mouseState.x, canvas.rows), CV_RGB(255, 0, 0), 1);
        } else {
            // black
            cv::rectangle(canvas, cv::Point(boundingBox.x, boundingBox.y),
                cv::Point(mouseState.x, mouseState.y), CV_RGB(0, 0, 0), 2);
            // red
            cv::rectangle(canvas, cv::Point(boundingBox.x, boundingBox.y),
                cv::Point(mouseState.x, mouseState.y), CV_RGB(255, 0, 0), 1);
        }

        cv::imshow(windowName, canvas);

        // wait for mouse click (or key press)
        while (mouseState.event == -1) {
            int ch = cv::waitKey(30);
            if (ch != -1) break;
        }

        // check event type
        if (mouseState.event == -1) {
            break;
        } else if (mouseState.event == CV_EVENT_LBUTTONUP) {
            if (clickCount == 0) {
                boundingBox.x = mouseState.x;
                boundingBox.y = mouseState.y;
            } else {
                boundingBox.width = mouseState.x - boundingBox.x + 1;
                boundingBox.height = mouseState.y - boundingBox.y + 1;
            }
            clickCount += 1;
        }

        mouseState.event = -1;
    }

    cv::setMouseCallback(windowName, NULL);
    if (bDestoryWindow) {
        cv::destroyWindow(windowName);
    }

    return boundingBox;
}

cv::Mat drwnInputScribble(const string& windowName, const cv::Mat& img,
    const cv::Scalar& colour, int width)
{
    DRWN_ASSERT((img.data != NULL) && (width > 0));
    drwnMouseState mouseState;

    // initialize state
    bool bDestoryWindow = (cvGetWindowHandle(windowName.c_str()) == NULL);
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, drwnOnMouse, (void *)&mouseState);

    cv::Mat canvas(img.rows, img.cols, CV_8UC3);
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    // event processing loop
    cv::Point lastPoint(0, 0);
    while (1) {
        // update canvas
        img.copyTo(canvas);
        canvas.setTo(colour, mask);
        cv::imshow(windowName, canvas);

        // wait for next event
        while (mouseState.event == -1) {
            int ch = cv::waitKey(30);
            if (ch != -1) break;
        }

        // check event type
        if (mouseState.event == -1) {
            break;
        } else if (mouseState.event == CV_EVENT_MOUSEMOVE) {
            if ((mouseState.flags & CV_EVENT_FLAG_LBUTTON) != 0) {
                cv::line(mask, cv::Point(mouseState.x, mouseState.y), lastPoint, cv::Scalar(1), width);
            }
        }

        lastPoint = cv::Point(mouseState.x, mouseState.y);
        mouseState.event = -1;
    }

    // restore state
    cv::setMouseCallback(windowName, NULL);
    if (bDestoryWindow) {
        cv::destroyWindow(windowName);
    }

    return mask;
}

double drwnComparePatches(const cv::Mat& patchA, const cv::Mat& patchB, int method)
{
    DRWN_ASSERT((patchA.data != NULL) && (patchB.data != NULL));

    switch (method) {
    case CV_TM_SQDIFF:
        {
            const double d = cv::norm(patchA - patchB);
            return d * d;
        }
        break;
    case CV_TM_SQDIFF_NORMED:
    case CV_TM_CCORR:
    case CV_TM_CCORR_NORMED:
        {
            const double a = cv::norm(patchA);
            const double b = cv::norm(patchB);
            if ((a == 0.0) || (b == 0.0))
                return 0.0;

            double d = cv::norm(patchA - patchB);
            if ((method == CV_TM_CCORR) || (method == CV_TM_CCORR_NORMED)) {
                d = 0.5 * (a * a + b * b - d);
            }

            if (method == CV_TM_SQDIFF_NORMED)
                return (d / a) * (d / b);
            else if (method == CV_TM_CCORR)
                return d;
            else return d / (a * b);
        }
        break;
    case CV_TM_CCOEFF:
    case CV_TM_CCOEFF_NORMED:
        DRWN_TODO;
        break;
    default:
        DRWN_LOG_FATAL("unrecognized method " << method);
    }

    return 0.0;
}

// Drawing functions
void drwnOverlayImages(cv::Mat& canvas, const cv::Mat& overlay, double alpha)
{
    DRWN_ASSERT((canvas.data != NULL) && (overlay.data != NULL));
    DRWN_ASSERT((canvas.rows == overlay.rows) && (canvas.cols == overlay.cols));
    alpha = std::min(1.0, std::max(0.0, alpha));

    drwnColorImageInplace(canvas);
    cv::Mat tmp = drwnColorImage(overlay);
    cv::addWeighted(canvas, (1.0 - alpha), tmp, alpha, 0.0, canvas);
}

void drwnOverlayMask(cv::Mat& canvas, const cv::Mat& mask, const cv::Scalar& color, double alpha)
{
    DRWN_ASSERT((canvas.data != NULL) && (mask.data != NULL));
    DRWN_ASSERT((canvas.rows == mask.rows) && (canvas.cols == mask.cols));

    cv::Mat w(mask.rows, mask.cols, CV_32FC1);
    mask.convertTo(w, CV_32F);
    w = cv::min(cv::max(w, 0.0), 1.0);

    vector<cv::Mat> channels(3);
    w.convertTo(channels[0], CV_8U, color[0]);
    w.convertTo(channels[1], CV_8U, color[1]);
    w.convertTo(channels[2], CV_8U, color[2]);

    cv::Mat overlay;
    cv::merge(channels, overlay);

    drwnOverlayImages(canvas, overlay, alpha);
}

void drwnShadeRectangle(cv::Mat& canvas, cv::Rect roi, const cv::Scalar& color,
    double alpha, drwnFillType fill, int thickness)
{
    DRWN_ASSERT(canvas.data != NULL);
    drwnTruncateRect(roi, canvas.cols, canvas.rows);
    if ((roi.width == 0) || (roi.height == 0))
        return;

    cv::Mat mask = cv::Mat::zeros(canvas.rows, canvas.cols, CV_8UC1);
    mask(roi).setTo(cv::Scalar(1));

    drwnShadeRegion(canvas, mask, color, alpha, fill, thickness);
}

void drwnShadeRegion(cv::Mat& canvas, const cv::Mat& mask, const cv::Scalar& color,
    double alpha, drwnFillType fill, int thickness)
{
    DRWN_ASSERT((canvas.data != NULL) && (mask.data != NULL));
    DRWN_ASSERT(mask.type() == CV_8UC1);

    alpha = std::min(1.0, std::max(0.0, alpha));
    thickness = std::max(1, thickness);
    drwnColorImageInplace(canvas);

    for (int y = 0; y < canvas.rows; y++) {
        const unsigned char *m = mask.ptr<const unsigned char>(y);
        unsigned char * const p = canvas.ptr<unsigned char>(y);
        for (int x = 0; x < canvas.cols; x++) {
            if (m[x] == 0x00) continue;

            switch (fill) {
            case DRWN_FILL_SOLID:
                p[3 * x + 2] = (unsigned char)((1.0 - alpha) * p[3 * x + 2] + alpha * color.val[2]);
                p[3 * x + 1] = (unsigned char)((1.0 - alpha) * p[3 * x + 1] + alpha * color.val[1]);
                p[3 * x + 0] = (unsigned char)((1.0 - alpha) * p[3 * x + 0] + alpha * color.val[0]);
                break;
            case DRWN_FILL_DIAG:
                if ((int(x / thickness) + int(y / thickness)) % 4 == 0) {
                    p[3 * x + 2] = (unsigned char)((1.0 - alpha) * p[3 * x + 2] + alpha * color.val[2]);
                    p[3 * x + 1] = (unsigned char)((1.0 - alpha) * p[3 * x + 1] + alpha * color.val[1]);
                    p[3 * x + 0] = (unsigned char)((1.0 - alpha) * p[3 * x + 0] + alpha * color.val[0]);
                }
                break;
            case DRWN_FILL_CROSSHATCH:
                if (((int(x / thickness) + int(y / thickness)) % 4 == 0) ||
                    ((int(x / thickness) + canvas.rows - int(y / thickness)) % 4 == 0)) {
                    p[3 * x + 2] = (unsigned char)((1.0 - alpha) * p[3 * x + 2] + alpha * color.val[2]);
                    p[3 * x + 1] = (unsigned char)((1.0 - alpha) * p[3 * x + 1] + alpha * color.val[1]);
                    p[3 * x + 0] = (unsigned char)((1.0 - alpha) * p[3 * x + 0] + alpha * color.val[0]);
                }
                break;
            }
        }
    }
}

void drwnMaskRegion(cv::Mat& canvas, const cv::Mat& mask)
{
    DRWN_ASSERT((canvas.data != NULL) && (mask.data != NULL));
    DRWN_ASSERT((canvas.rows == mask.rows) && (canvas.cols == mask.cols));

    cv::Mat tmp(mask.rows, mask.cols, CV_32FC1);
    vector<cv::Mat> channels(canvas.channels());
    cv::split(canvas, &channels[0]);
    for (unsigned c = 0; c < channels.size(); c++) {
        channels[c].convertTo(tmp, CV_32F);
        tmp *= mask;
        tmp.convertTo(channels[c], channels[c].depth());
    }
    cv::merge(&channels[0], channels.size(), canvas);
}

void drwnDrawRegionBoundaries(cv::Mat& canvas, const cv::Mat& mask,
    const cv::Scalar& color, int thickness)
{
    DRWN_ASSERT((canvas.data != NULL) && (mask.data != NULL));
    DRWN_ASSERT((canvas.rows == mask.rows) && (canvas.cols == mask.cols));
    DRWN_ASSERT(mask.channels() == 1);

    switch (mask.depth()) {
    case CV_8U:
    case CV_8S:
    case CV_16S:
        {
            cv::Mat m(mask.rows, mask.cols, CV_32S);
            mask.convertTo(m, CV_32S);
            drwnDrawRegionBoundaries(canvas, m, color, thickness);
            break;
        }
    case CV_32S:
        {
            // create boundary mask
            cv::Mat boundaryMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8U);

            // find boundaries
            for (int y = 0; y < canvas.rows; y++) {
                const int *m = (const int *)mask.ptr<int>(y);
                unsigned char * const p = (unsigned char *)boundaryMask.ptr<unsigned char>(y);
                for (int x = 0; x < canvas.cols; x++) {
                    // check 4-neighbours
                    if (((x > 0) && (m[x] < m[x - 1])) ||
                        ((x < canvas.cols - 1) && (m[x] < m[x + 1])) ||
                        ((y > 0) && (m[x] < m[x - canvas.cols])) ||
                        ((y < canvas.rows - 1) && (m[x] < m[x + canvas.cols]))) {
                        p[x] = 0x01;
                    }
                }
            }

            // dialate boundary mask
            if (thickness > 1) {
                cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(thickness, thickness));
                cv::dilate(boundaryMask, boundaryMask, kernel);
            }

            // colour boundary
            canvas.setTo(color, boundaryMask);

            break;
        }
    default:
        DRWN_LOG_FATAL("mask must be an integer type");
    }
}

void drwnDrawRegionBoundary(cv::Mat& canvas, const cv::Mat& mask,
    int idRegionA, int idRegionB, const cv::Scalar& color, int thickness)
{
    DRWN_ASSERT((canvas.data != NULL) && (mask.data != NULL));
    DRWN_ASSERT(canvas.size() == mask.size());
    DRWN_ASSERT(mask.channels() == 1);

    switch (mask.depth()) {
    case CV_8U:
    case CV_8S:
    case CV_16S:
        {
            cv::Mat m(mask.rows, mask.cols, CV_32SC1);
            mask.convertTo(m, CV_32S);
            drwnDrawRegionBoundaries(canvas, m, color, thickness);
            break;
        }
    case CV_32S:
        {
            // create boundary mask
            cv::Mat boundaryMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);

            // find boundary pixels
            for (int y = 0; y < canvas.rows; y++) {
                const int *m = mask.ptr<const int>(y);
                unsigned char * const p = boundaryMask.ptr<unsigned char>(y);
                for (int x = 0; x < canvas.cols; x++) {
                    // check 4-neighbours (draw on idRegionA side)
                    if (((x > 0) && (m[x] == idRegionA) && (m[x - 1] == idRegionB)) ||
                        ((x < canvas.cols - 1) && (m[x] == idRegionA) && (m[x + 1] == idRegionB)) ||
                        ((y > 0) && (m[x] == idRegionA) && (m[x - canvas.cols] == idRegionB)) ||
                        ((y < canvas.rows - 1) && (m[x] == idRegionA) && (m[x + canvas.cols] == idRegionB))) {
                        p[x] = 0x01;
                    }
                }
            }

            // dialate boundary mask
            if (thickness > 1) {
                cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(thickness, thickness));
                cv::dilate(boundaryMask, boundaryMask, kernel);
            }

            // colour boundary
            canvas.setTo(color, boundaryMask);

            break;
        }
    default:
        DRWN_LOG_FATAL("mask must be an integer type");
    }
}

void drwnAverageRegions(cv::Mat& img, const cv::Mat& seg)
{
    DRWN_ASSERT((img.data != NULL) && (seg.data != NULL));
    DRWN_ASSERT((img.rows == seg.rows) && (img.cols == seg.cols));
    DRWN_ASSERT((img.depth() == CV_32F) || (img.depth() == CV_8U));
    DRWN_ASSERT((seg.channels() == 1) && (seg.depth() == CV_32S));

    // accumulate channel sums
    map<int, pair<unsigned, vector<double> > > suffstats;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {

            // find sufficient statistics for current segment id
            const int segId = seg.at<int>(y, x);
            map<int, pair<unsigned, vector<double> > >::iterator it = suffstats.find(segId);
            if (it == suffstats.end()) {
                it = suffstats.insert(it, make_pair(segId,
                        make_pair(unsigned(0), vector<double>(img.channels(), 0.0))));
            }

            // update the sufficient statistics
            if (img.depth() == CV_8U) {
                for (int c = 0; c < img.channels(); c++) {
                    it->second.second[c] += (double)img.at<unsigned char>(y, x * img.channels() + c);
                }
            } else {
                for (int c = 0; c < img.channels(); c++) {
                    it->second.second[c] += (double)img.at<float>(y, x * img.channels() + c);
                }
            }
            it->second.first += 1;
        }
    }

    // create averages
    for (map<int, pair<unsigned, vector<double> > >::iterator it = suffstats.begin(); it != suffstats.end(); ++it) {
        for (vector<double>::iterator jt = it->second.second.begin(); jt != it->second.second.end(); ++jt) {
            *jt /= (double)it->second.first;
        }
    }

    // copy averaged values back to image
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            const int segId = seg.at<int>(y, x);
            map<int, pair<unsigned, vector<double> > >::const_iterator it = suffstats.find(segId);
            if (img.depth() == CV_8U) {
                for (int c = 0; c < img.channels(); c++) {
                    img.at<unsigned char>(y, x * img.channels() + c) = (unsigned char)it->second.second[c];
                }
            } else {
                for (int c = 0; c < img.channels(); c++) {
                    img.at<float>(y, x * img.channels() + c) = (float)it->second.second[c];
                }
            }
        }
    }
}

// drwnOpenCVUtilsConfig ----------------------------------------------------
//! \addtogroup drwnConfigSettings
//! \section drwnOpenCVUtils
//! \b maxShowHeight :: maximum height for displaying images\n
//! \b maxShowWidth  :: maximum width for displaying images\n

class drwnOpenCVUtilsConfig : public drwnConfigurableModule {
public:
    drwnOpenCVUtilsConfig() : drwnConfigurableModule("drwnOpenCVUtils") { }
    ~drwnOpenCVUtilsConfig() { }

    void usage(ostream &os) const {
        os << "      maxShowHeight   :: maximum height for displaying images\n";
        os << "      maxShowWidth    :: maximum width for displaying images\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxShowHeight")) {
            drwnOpenCVUtils::SHOW_IMAGE_MAX_HEIGHT = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxShowWidth")) {
            drwnOpenCVUtils::SHOW_IMAGE_MAX_WIDTH = std::max(1, atoi(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnOpenCVUtilsConfig gOpenCVUtilsConfig;

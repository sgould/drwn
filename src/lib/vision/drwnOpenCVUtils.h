/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnOpenCVUtils.h
** \anchor drwnOpenCVUtils
** \brief Utilities for working with OpenCV.
**
** The testDarwinVision application demonstrates how to use many of these
** functions.
*/

#pragma once

#include <cstdlib>
#include <vector>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"

using namespace std;
using namespace Eigen;

// Global parameters that can be set from the command line
namespace drwnOpenCVUtils {
    //! maximum image height for drwnShowDebuggingImage
    extern unsigned SHOW_IMAGE_MAX_HEIGHT;
    //! maximum image width for drwnShowDebuggingImage
    extern unsigned SHOW_IMAGE_MAX_WIDTH;
};

// Convert opencv data structures to a string
string toString(const cv::Mat& m);
string toString(const vector<cv::Mat>& vm);
string toString(const cv::Rect& r);
string toString(const cv::Point& pt);
string toString(const cv::Size& sz);
string toString(const cv::Scalar& slr);

// Operators
//! equality operator for CvRect objects
bool operator==(const CvRect& r, const CvRect& s);
//! inequality operator for CvSize objects (allows partial sorting)
bool operator<(const CvSize& r, const CvSize& s);
//! inequality operator for CvPoint objects (allows partial sorting)
bool operator<(const CvPoint& p, const CvPoint& q);

//! count number of entries matching comparison
int drwnCmpCount(const cv::Mat& s, const cv::Mat& t, int cmpOp = CV_CMP_EQ);

//! show an image (scale if not CV_8U) and wait (returns result from cv::waitKey)
int drwnShowDebuggingImage(const cv::Mat& img, const std::string& name, bool bWait);
//! show an array of images and wait (returns result from cv::waitKey)
int drwnShowDebuggingImage(const vector<cv::Mat>& views, const std::string& name, bool bWait, int rows = -1);

//! finds the smallest bounding box around these rectangles
cv::Rect drwnFitBoundingBox(const vector<cv::Rect> &rects);
//! finds the smallest bounding box around these points
cv::Rect drwnFitBoundingBox(const vector<cv::Point> &points);
//! finds the smallest bounding box around these rectangles
//! with given aspect ratio (w/h)
cv::Rect drwnFitBoundingBox(cv::Rect r, double aspectRatio);

//! area defined by the rectangle
inline double area(const cv::Rect& r) { return (double)r.area(); }
//! aspect ratio defined by the rectangle
inline double aspect(const cv::Rect& r) { return (double)r.width / (double)r.height; }
//! area of the intersection between two rectangles
double areaOverlap(const cv::Rect& r, const cv::Rect& s);
//! area of the union between two rectangles
inline double areaUnion(const cv::Rect& r, const cv::Rect& s) {
    return area(r) + area(s) - areaOverlap(r, s);
}

//! convert image to greyscale (32-bit floating point)
cv::Mat drwnGreyImage(const cv::Mat& src);
//! convert image to color (8-bit)
cv::Mat drwnColorImage(const cv::Mat& src);
//! compute a soft edge map for an image (32-bit floating point)
cv::Mat drwnSoftEdgeMap(const cv::Mat& src, bool bNormalize = false);
//! convert an image to 32-bit greyscale in place
void drwnGreyImageInplace(cv::Mat& img);
//! convert an image to color in place
void drwnColorImageInplace(cv::Mat& img);

//! Compute the pixelwise average of a stack of images. All images
//! must be of the same size and type.
cv::Mat drwnPixelwiseMean(const vector<cv::Mat>& imgStack);
//! Compute the pixelwise median of a stack of images. All images
//! must be of the same size and of type CV_8U. The images must
//! also be continuous.
cv::Mat drwnPixelwiseMedian(const vector<cv::Mat>& imgStack);

//! pad an image and copy the boundary
cv::Mat drwnPadImage(const cv::Mat& src, int margin);
//! pad an image and copy the boundary, top-left reference is (0, 0) so
//! page x- and y-coordinates must be non-positive
cv::Mat drwnPadImage(const cv::Mat& src, const cv::Rect& page);

//! translate an array
cv::Mat drwnTranslateMatrix(const cv::Mat& matrix, const cv::Point& origin,
    double fillValue = 0.0);

//! rotate an image clockwise by \p theta
cv::Mat drwnRotateImage(const cv::Mat& img, float theta);

//! scale all entries in the image/matrix to the given range
void drwnScaleToRange(cv::Mat& m, double minValue = 0.0, double maxValue = 1.0);
//! resize an image in place
void drwnResizeInPlace(cv::Mat& m, const cv::Size& size, int interpolation = CV_INTER_LINEAR);
//! resize a matrix in place
void drwnResizeInPlace(cv::Mat& m, int rows, int cols, int interpolation = CV_INTER_LINEAR);
//! crop an image or matrix in place
void drwnCropInPlace(cv::Mat& image, cv::Rect roi);

//! returns true if the rectangle has non-zero size and fits within the given image dimensions
bool drwnValidRect(const cv::Rect& r, int width, int height);
//! returns true if the rectangle has non-zero size and fits within the given image
inline bool drwnValidRect(const cv::Rect& r, const cv::Mat& img) { return drwnValidRect(r, img.cols, img.rows); }
//! truncates a rectangle to fit inside [0, 0, width - 1, height - 1]
void drwnTruncateRect(cv::Rect& r, int width, int height);
//! truncates a rectangle to fit inside the image
inline void drwnTruncateRect(cv::Rect& r, const cv::Mat& img) { return drwnTruncateRect(r, img.cols, img.rows); }

//! Assemble images into one big image. All images must be of the same format
//! and \p rows * \p cols must be smaller than \p images.size(). If negative then
//! will choose a square (rows = cols = ceil(sqrt(images.size()))). A border can
//! be added around each image of given \p margin and \p colour.
cv::Mat drwnCombineImages(const vector<cv::Mat>& images, int rows = -1, int cols = -1,
    unsigned margin = 0, const cv::Scalar& colour = cv::Scalar(0));

typedef enum {
    DRWN_COLORMAP_RAINBOW,  //!< ranges from blue to red through green and yellow
    DRWN_COLORMAP_HOT,      //!< ranges from red to yellow
    DRWN_COLORMAP_COOL,     //!< ranges from cyan to pink
    DRWN_COLORMAP_REDGREEN, //!< ranges from green to red
    DRWN_COLORMAP_ANU       //!< ranges over ANU corporate colours
} drwnColorMap;

//! Convert a floating point matrix with entries in range [0, 1] to color
//! image heatmap in either rainbow, hot or cool specturms. Integer matrices
//! are first rescaled.
cv::Mat drwnCreateHeatMap(const cv::Mat& m, drwnColorMap cm = DRWN_COLORMAP_RAINBOW);

//! Convert a floating point matrix with entries in range [0, 1] to color
//! image heatmap that interpolates between the given two colours.
cv::Mat drwnCreateHeatMap(const cv::Mat& m, cv::Scalar colourA, cv::Scalar colourB);

//! Convert a floating point matrix with entries in range [0, 1] to color
//! image heatmap that interpolates between the given colour table.
cv::Mat drwnCreateHeatMap(const cv::Mat& m, const vector<cv::Scalar>& colours);

//! Creates a color table by uniformly sampling a colormap.
vector<cv::Scalar> drwnCreateColorTable(unsigned n, drwnColorMap cm = DRWN_COLORMAP_RAINBOW);

//! draws a pretty bounding box
void drwnDrawBoundingBox(cv::Mat& canvas, const cv::Rect& roi, cv::Scalar fgcolor,
    cv::Scalar bgcolor = CV_RGB(255, 255, 255), int lineWidth = 2);

//! draws a pretty polygon
void drwnDrawPolygon(cv::Mat& canvas, const vector<cv::Point> &poly, cv::Scalar fgcolor,
    cv::Scalar bgcolor = CV_RGB(255, 255, 255), int lineWidth = 2, bool bClose = true);

//! draws a line across the image, optionally filling below the line
void drwnDrawFullLinePlot(cv::Mat& canvas, const vector<double>& points,
    const cv::Scalar& lineColour, unsigned lineWidth = 2,
    const cv::Scalar& baseShading = CV_RGB(0, 0, 0), double baseAlpha = 0.0);

//! draws a target symbol (circle and cross-hairs)
void drwnDrawTarget(cv::Mat& canvas, const cv::Point& center,
    cv::Scalar color = CV_RGB(255, 0, 0), int size = 5, int lineWidth = 1);

//! Mouse state and mouse callback for populating the mouse state. Used by the
//! \ref drwnWaitMouse function.
class drwnMouseState {
 public:
    int event, x, y, flags;

 public:
    drwnMouseState() : event(-1), x(0), y(0), flags(0) { /* do nothing */ }
    ~drwnMouseState() { /* do nothing */ }
};

//! Mouse callback function (populates \ref MouseState data members passed
//! as a pointer to void).
void drwnOnMouse(int event, int x, int y, int flags, void *ptr);

//! Waits for up to \p numPoints mouse clicks (or key press) and returns the location
//! of the points. The clicked points are indicated on the given image. The window
//! be destroyed on return unless already open. The function is similar to Matlab's
//! ginput() function.
vector<cv::Point> drwnWaitMouse(const string& windowName, const cv::Mat& img,
    int numPoints = DRWN_INT_MAX);

//! Waits for the user to input a bounding box by clicking two points on a canvas.
cv::Rect drwnInputBoundingBox(const string& windowName, const cv::Mat& img);

//! Allows a user to scribble on a canvas. Exits when they press a key.
cv::Mat drwnInputScribble(const string& windowName, const cv::Mat& img,
    const cv::Scalar& colour = CV_RGB(255, 0, 0), int width = 5);

//! Compute distance between two image patches. \p method can be one of
//! CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCORR, CV_TM_CCORR_NORMED,
//! CV_TM_CCOEFF, or CV_TM_CCOEFF_NORMED.
double drwnComparePatches(const cv::Mat& patchA, const cv::Mat& patchB, int method = CV_TM_SQDIFF);

// Drawing functions
typedef enum {
    DRWN_FILL_SOLID,      //!< fill with solid color
    DRWN_FILL_DIAG,       //!< fill with diagonal stripes
    DRWN_FILL_CROSSHATCH  //!< fill with crosshatching
} drwnFillType;

//! overlays an image on the (same size) canvas
void drwnOverlayImages(cv::Mat& canvas, const cv::Mat& overlay, double alpha = 0.5);
//! overlays a soft mask (in range [0, 1]) on the (same size) canvas
void drwnOverlayMask(cv::Mat& canvas, const cv::Mat& mask, const cv::Scalar& color, double alpha = 0.5);
//! draws a shaded rectangle on the canvas
void drwnShadeRectangle(cv::Mat& canvas, cv::Rect roi, const cv::Scalar& color,
    double alpha = 0.5, drwnFillType fill = DRWN_FILL_SOLID, int thickness = 1);
//! draws a shaded region on the canvas
void drwnShadeRegion(cv::Mat& canvas, const cv::Mat& mask, const cv::Scalar& color,
    double alpha = 0.5, drwnFillType fill = DRWN_FILL_SOLID, int thickness = 1);
//! applies a soft mask (in range [0, 1]) to a region on the canvas
void drwnMaskRegion(cv::Mat& canvas, const cv::Mat& mask);
//! marks the boundary between regions
void drwnDrawRegionBoundaries(cv::Mat& canvas, const cv::Mat& mask,
    const cv::Scalar& color, int thickness = 1);
//! marks the boundary between two specific regions
void drwnDrawRegionBoundary(cv::Mat& canvas, const cv::Mat& mask,
    int idRegionA, int idRegionB, const cv::Scalar& color, int thickness = 1);
//! Fills each region with the average of the colour within the region.
//! \p img can have an arbitrary number of channels.
//! \p seg must be the same size as the image and of type CV_32SC1
void drwnAverageRegions(cv::Mat& img, const cv::Mat& seg);

/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImageInPainter.cpp
** AUTHOR(S):   Robin Liang <robin.gnail@gmail.com>
**              Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// eigen matrix library headers
#include "Eigen/Core"

// opencv library headers
#include "cv.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// drwnImageInPainter --------------------------------------------------------

unsigned drwnImageInPainter::UPDATE_STEPS = 10;
bool drwnImageInPainter::PIXELWISE = false;
bool drwnImageInPainter::PRIORITY_FILLING = false;

cv::Mat drwnImageInPainter::fill(const cv::Mat& image, const cv::Mat& fillMask,
    const cv::Mat& copyMask, const cv::Mat& ignoreMask) const
{
    DRWN_ASSERT((patchRadius > 0) && ((unsigned)image.rows > 2 * patchRadius + 1) &&
        ((unsigned)image.cols > 2 * patchRadius + 1));
    DRWN_FCN_TIC;

    // working images and masks
    cv::Mat workingImage = image.clone();
    cv::Mat workingMask = (fillMask == 0x00);

    // featurize the image
    cv::Mat fImage = featurizeImage(workingImage);

    // start inpainting
    drwnMaskedPatchMatch pm(fImage, fImage, ignoreMask == 0x00, copyMask & workingMask, patchRadius);
    bool bKeepUpdating = true;
    while (bKeepUpdating) {
        DRWN_PROGRESS_SPINNER("Inpainting with patch size " << (2 * patchRadius + 1) << "-by-" << (2 * patchRadius + 1) << "...")
        bKeepUpdating = false;
        pm.search(UPDATE_STEPS);

        // check 8-neighbour boundary
        cv::Mat boundary = extractMaskBoundary(workingMask);

        if (PRIORITY_FILLING) {
            // find boundary pixel with highest priority
            vector<pair<double, pair<int, int> > > priorities;
            for (int y = 0; y < workingImage.rows; y++) {
                for (int x = 0; x < workingImage.cols; x++) {
                    if (boundary.at<unsigned char>(y, x) != 0xff)
                        continue;

                    const cv::Point p(x, y);
                    double d = computePixelPriority(p, workingImage, workingMask);
                    priorities.push_back(make_pair(d, make_pair(x, y)));
                }
            }
            if (!priorities.empty()) {
                std::sort(priorities.begin(), priorities.end());
                boundary.setTo(cv::Scalar(0x00));
                const pair<int, int> p = priorities[0].second;
                boundary.at<unsigned char>(p.second, p.first) = 0xff;
            }
        }

        // inpaint boundary pixels
        for (int y = 0; y < workingImage.rows; y++) {
            for (int x = 0; x < workingImage.cols; x++) {
                if (boundary.at<unsigned char>(y, x) == 0xff) {
                    bKeepUpdating = true;

                    const cv::Point p(x, y);
                    const pair<cv::Rect, cv::Rect> match = pm.getMatchingPatches(p);

                    if (PIXELWISE) {

                        const cv::Point q(match.second.x + match.second.width / 2,
                            match.second.y + match.second.height / 2);
                        pm.modifySourceImage(cv::Rect(x, y, 1, 1), cv::Rect(q.x, q.y, 1, 1));

                        // allow copying from within infilled region
                        if (bAllowFromFilled) {
                            pm.modifyTargetImage(cv::Rect(x, y, 1, 1), cv::Rect(q.x, q.y, 1, 1));
                        }

                        // inpaint corresponding region on large image
                        cv::Rect srcROI(x, y, 1, 1);
                        drwnTruncateRect(srcROI, workingImage.cols, workingImage.rows);
                        cv::Rect tgtROI(q.x + x - srcROI.x, q.y + y - srcROI.y, srcROI.width, srcROI.height);
                        drwnTruncateRect(tgtROI, workingImage.cols, workingImage.rows);
                        srcROI.width = tgtROI.width; srcROI.height = tgtROI.height;
                        DRWN_ASSERT(srcROI.area() > 0);
                        workingImage(tgtROI).copyTo(workingImage(srcROI));
                        workingMask(srcROI).setTo(cv::Scalar(0xff));

                    } else {

                        pm.modifySourceImage(match.first, match.second, alphaBlend);

                        // allow copying from within infilled region
                        if (bAllowFromFilled) {
                            pm.modifyTargetImage(match.first, match.second, alphaBlend);
                        }

                        // inpaint corresponding region on large image
                        cv::Rect srcROI(match.first.x, match.first.y, match.first.width, match.first.height);
                        drwnTruncateRect(srcROI, workingImage.cols, workingImage.rows);
                        cv::Rect tgtROI(match.second.x + match.first.x - srcROI.x,
                            match.second.y + match.first.y - srcROI.y,
                            srcROI.width, srcROI.height);
                        drwnTruncateRect(tgtROI, workingImage.cols, workingImage.rows);
                        srcROI.width = tgtROI.width; srcROI.height = tgtROI.height;
                        DRWN_ASSERT(srcROI.area() > 0);
                        //workingImage(tgtROI).copyTo(workingImage(srcROI));
                        blendMaskedImage(workingImage(srcROI), workingImage(tgtROI), workingMask(srcROI), alphaBlend);
                        workingMask(srcROI).setTo(cv::Scalar(0xff));
                    }
                }
            }
        }

        // show progress
        if (bVisualize) {
            monitor(workingImage, workingMask);
            drwnShowDebuggingImage(pm.visualize(), "pm", false);
        }
    }

    // return inpainted image
    DRWN_FCN_TOC;
    return workingImage;
}

void drwnImageInPainter::monitor(const cv::Mat& image, const cv::Mat& mask) const
{
    cv::Mat canvas = image.clone();
    drwnDrawRegionBoundaries(canvas, mask, CV_RGB(0, 255, 0), 2);
    drwnShowDebuggingImage(canvas, "drwnImageInPainter", false);
}

cv::Mat drwnImageInPainter::featurizeImage(const cv::Mat& img) const
{
    cv::Mat pixfeats = cv::Mat(img.rows, img.cols, CV_8UC(4), cv::Scalar(0));

    // Lab colour features
    cv::Mat lab(img.rows, img.cols, CV_8UC3);
    cv::cvtColor(img, lab, CV_BGR2Lab);
    int from_to[] = {0, 0};
    for (int c = 0; c < 3; c++) {
        from_to[0] = c; from_to[1] = c;
        cv::mixChannels(&lab, 1, &pixfeats, 1, from_to, 1);
    }

    // edge map features
    cv::Mat edges = drwnSoftEdgeMap(img, true);
    cv::Mat edgesU8(img.rows, img.cols, CV_8UC1);
    edges.convertTo(edgesU8, CV_8U, 255.0, 0.0);
    from_to[0] = 0; from_to[1] = 3;
    cv::mixChannels(&edgesU8, 1, &pixfeats, 1, from_to, 1);

    return pixfeats;
}

double drwnImageInPainter::computePixelPriority(const cv::Point& p,
    const cv::Mat& image, const cv::Mat& mask) const
{
    cv::Rect roi(p.x - 3, p.y - 3, 7, 7);
    //cv::Rect roi(p.x - 7, p.y - 7, 15, 15);
    drwnTruncateRect(roi, image);

    cv::Mat grey;
    cv::cvtColor(image(roi), grey, CV_RGB2GRAY);

    cv::Mat g;
    cv::Sobel(grey, g, CV_16S, 1, 0);
    double dxImg = cv::mean(g, mask(roi))[0];
    cv::Sobel(grey, g, CV_16S, 0, 1);
    double dyImg = cv::mean(g, mask(roi))[0];

    cv::Sobel(mask(roi), g, CV_16S, 1, 0);
    double dxMsk = cv::mean(g)[0];
    cv::Sobel(mask(roi), g, CV_16S, 0, 1);
    double dyMsk = cv::mean(g)[0];

    double d = dxImg * dxImg + dyImg * dyImg + DRWN_EPSILON;
    dxImg /= d; dyImg /= d;

    d = dxMsk * dxMsk + dyMsk * dyMsk + DRWN_EPSILON;
    dxMsk /= d; dyMsk /= d;

    return fabs(dyMsk * dyImg + dxMsk * dxImg);
}

cv::Mat drwnImageInPainter::extractMaskBoundary(const cv::Mat& mask)
{
    cv::Mat boundary = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);

    for (int y = 0; y < mask.rows; y++) {
        const unsigned char *m = mask.ptr<const unsigned char>(y);
        unsigned char * const b = boundary.ptr<unsigned char>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (m[x] != 0x00) continue;
            if (((x > 0) && (m[x - 1] != 0x00)) ||
                ((x < mask.cols - 1) && (m[x + 1] != 0x00)) ||
                ((y > 0) && (m[x - mask.cols] != 0x00)) ||
                ((y < mask.rows - 1) && (m[x + mask.cols] != 0x00)) ||
                ((x > 0) && (y > 0) && (m[x - mask.cols - 1] != 0x00)) ||
                ((x > 0) && (y < mask.rows - 1) && (m[x + mask.cols - 1] != 0x00)) ||
                ((x < mask.cols - 1) && (y > 0) && (m[x - mask.cols + 1] != 0x00)) ||
                ((x < mask.cols - 1) && (y < mask.rows - 1) && (m[x + mask.cols + 1] != 0x00))) {
                b[x] = 0xff;
            }
        }
    }

    return boundary;
}

void drwnImageInPainter::blendMaskedImage(cv::Mat img, const cv::Mat& src, const cv::Mat& mask, double alpha)
{
    DRWN_ASSERT((src.size() == img.size()) && (mask.size() == img.size()));
    DRWN_ASSERT((src.type() == img.type()) && (mask.type() == CV_8UC1));
    DRWN_ASSERT((0.0 <= alpha) && (alpha <= 1.0));

    src.copyTo(img, mask == 0x00);
    if (alpha != 0.0) {
        cv::addWeighted(img, (1.0 - alpha), src, alpha, 0.0, img);
    }
}

// drwnImageInPainterConfig -------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnGrabCut
//! \b updateSteps :: number of PatchMatch search steps between updates (default: 10)\n
//! \b doPixelwise :: do pixelwise update rather than patchwise (default: false)\n
//! \b priorityFill :: do priority filling rather than onion filling (default: false)\n

class drwnImageInPainterConfig : public drwnConfigurableModule {
public:
    drwnImageInPainterConfig() : drwnConfigurableModule("drwnImageInPainter") { }
    ~drwnImageInPainterConfig() { }

    void usage(ostream &os) const {
        os << "      updateSteps     :: number of PatchMatch search steps between updates (default: "
           << drwnImageInPainter::UPDATE_STEPS << ")\n";
        os << "      doPixelwise     :: do pixelwise update rather than patchwise (default: "
           << (drwnImageInPainter::PIXELWISE ? "true" : "false") << ")\n";
        os << "      priorityFill    :: do priority filling rather than onion filling (default: "
           << (drwnImageInPainter::PRIORITY_FILLING ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "updateSteps")) {
            drwnImageInPainter::UPDATE_STEPS = std::max(1, atoi(value));
        } else if (!strcmp(name, "doPixelwise")) {
            drwnImageInPainter::PIXELWISE = drwn::trueString(string(value));
        } else if (!strcmp(name, "priorityFill")) {
            drwnImageInPainter::PRIORITY_FILLING = drwn::trueString(string(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnImageInPainterConfig gImageInPainterConfig;

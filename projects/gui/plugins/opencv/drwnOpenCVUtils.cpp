/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ Standard Libraries
#include <cstdlib>

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// OpenCV library
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

#include "drwnOpenCVUtils.h"

using namespace std;
using namespace Eigen;

// memory management
vector<CvMat *> createOpenCVMatrices(int nMats, int rows, int cols, int mType)
{
    DRWN_ASSERT(nMats > 0);
    vector<CvMat *> matrices(nMats);

    for (int i = 0; i < nMats; i++) {
        matrices[i] = cvCreateMat(rows, cols, mType);
        DRWN_ASSERT(matrices[i] != NULL);
    }

    return matrices;
}

void releaseOpenCVMatrices(vector<CvMat *>& matrices)
{
    for (int i = 0; i < (int)matrices.size(); i++) {
        if (matrices[i] != NULL) {
            cvReleaseMat(&matrices[i]);
            matrices[i] = NULL;
        }
    }
}

// type conversion
Eigen::MatrixXd cvMat2eigen(const CvMat *m)
{
    DRWN_ASSERT(m != NULL);

    MatrixXd d(m->rows, m->cols);

    switch (cvGetElemType(m)) {
    case CV_8UC1:
        {
            const unsigned char *p = (unsigned char *)CV_MAT_ELEM_PTR(*m, 0, 0);
            for (int i = 0; i < m->rows * m->cols; i++) {
                d[i] = (double)p[i];
            }
            d = d.transpose();
        }
        break;

    case CV_8SC1:
        {
            const char *p = (char *)CV_MAT_ELEM_PTR(*m, 0, 0);
            for (int i = 0; i < m->rows * m->cols; i++) {
                d[i] = (double)p[i];
            }
            d = d.transpose();
        }
        break;

    case CV_32SC1:
        d = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((int *)m->data.ptr, m->rows, m->cols).cast<double>();
        break;

    case CV_32FC1:
        d = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((float *)m->data.ptr, m->rows, m->cols).cast<double>();
        break;

    case CV_64FC1:
        d = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((double *)m->data.ptr, m->rows, m->cols);
        break;

    default:
        DRWN_LOG_FATAL("unrecognized openCV matrix type: " << cvGetElemType(m));
    }

    return d;
}

CvMat *eigen2cvMat(const Eigen::MatrixXd &m, int mType)
{
    CvMat *d = cvCreateMat(m.rows(), m.cols(), mType);
    DRWN_ASSERT(d != NULL);

    switch (mType) {
    case CV_8UC1:
    case CV_8SC1:
        DRWN_NOT_IMPLEMENTED_YET;
        break;

    case CV_32SC1:
        Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((int *)d->data.ptr, d->rows, d->cols) = m.cast<int>();
        break;

    case CV_32FC1:
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((float *)d->data.ptr, d->rows, d->cols) = m.cast<float>();
        break;

    case CV_64FC1:
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> >((double *)d->data.ptr, d->rows, d->cols) = m;
        break;

    default:
        DRWN_LOG_FATAL("unrecognized openCV matrix type: " << mType);
    }

    return d;
}

IplImage *record2Image(const drwnDataRecord *dataRec)
{
    if (dataRec == NULL)
        return NULL;

    // convert record to image
    DRWN_ASSERT(dataRec->structure().size() == 2);
    int height = dataRec->structure()[0];
    int width = dataRec->structure()[1];

    DRWN_ASSERT((dataRec->numFeatures() == 1) || (dataRec->numFeatures() == 3));
    DRWN_ASSERT(dataRec->numObservations() == width * height);

    bool bColour = (dataRec->numFeatures() == 3);
    IplImage *img = cvCreateImage(cvSize(width, height),
        IPL_DEPTH_8U, bColour ? 3 : 1);
    DRWN_ASSERT(img != NULL);

    int indx = 0;
    unsigned char *p = (unsigned char *)img->imageData;
    if (bColour) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++, indx++) {
                p[3 * x + 2] = (unsigned char)dataRec->data()(indx, 0);
                p[3 * x + 1] = (unsigned char)dataRec->data()(indx, 1);
                p[3 * x + 0] = (unsigned char)dataRec->data()(indx, 2);
            }
            p += img->widthStep;
        }
    } else {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++, indx++) {
                p[x] = (unsigned char)dataRec->data()[indx];
            }
            p += img->widthStep;
        }
    }

    return img;
}

vector<CvMat *> record2CvMats(const drwnDataRecord *dataRec)
{
    if (dataRec == NULL) return vector<CvMat *>();

    // convert record to vector of matrices
    DRWN_ASSERT(dataRec->structure().size() == 2);
    int rows = dataRec->structure()[0];
    int cols = dataRec->structure()[1];

    DRWN_ASSERT(dataRec->numObservations() == rows * cols);

    vector<CvMat *> m = createOpenCVMatrices(dataRec->numFeatures(), rows, cols, CV_64FC1);
    for (int i = 0; i < dataRec->numFeatures(); i++) {
        DRWN_ASSERT(m[i] != NULL);
        Eigen::Map<MatrixXd>((double *)m[i]->data.ptr, rows, cols) =
            dataRec->data().col(i);
    }

    return m;
}

void drwnClipRect(CvRect& r, int height, int width)
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

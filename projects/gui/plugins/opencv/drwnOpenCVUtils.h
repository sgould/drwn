/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOpenCVUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Utility functions for converting between OpenCV and Darwin datastructures.
**
*****************************************************************************/

#pragma once

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// OpenCV library
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

// memory management
vector<CvMat *> createOpenCVMatrices(int nMats, int rows, int cols, int mType);
void releaseOpenCVMatrices(vector<CvMat *>& matrices);

// type conversion
Eigen::MatrixXd cvMat2eigen(const CvMat *m);
CvMat *eigen2cvMat(const Eigen::MatrixXd &m, int mType = CV_32FC1);

IplImage *record2Image(const drwnDataRecord *dataRec);
//void image2Record(const IplImage *img, drwnDataRecord *dataRec);

vector<CvMat *> record2CvMats(const drwnDataRecord *dataRec);
//void cvMats2Record(const vector<CvMat *>& m, drwnDataRecord *dataRec);

// clipping and cropping
void drwnClipRect(CvRect& r, int height, int width);

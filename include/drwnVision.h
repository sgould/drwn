/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnVision.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Include file for computer vision headers.
**
*****************************************************************************/

#pragma once

#include "../src/lib/vision/drwnColourHistogram.h"
#include "../src/lib/vision/drwnFilterBankResponse.h"
#include "../src/lib/vision/drwnGrabCutInstance.h"
#include "../src/lib/vision/drwnHOGFeatures.h"
#include "../src/lib/vision/drwnImageCache.h"
#include "../src/lib/vision/drwnImageInPainter.h"
#include "../src/lib/vision/drwnImagePyramidCache.h"
#include "../src/lib/vision/drwnLBPFilterBank.h"
#include "../src/lib/vision/drwnMaskedPatchMatch.h"
#include "../src/lib/vision/drwnMultiSegConfig.h"
#include "../src/lib/vision/drwnMultiSegVis.h"
#include "../src/lib/vision/drwnObject.h"
#include "../src/lib/vision/drwnOpenCVUtils.h"
#include "../src/lib/vision/drwnPatchMatch.h"
#include "../src/lib/vision/drwnPatchMatchUtils.h"
#include "../src/lib/vision/drwnPartsModel.h"
#include "../src/lib/vision/drwnPixelNeighbourContrasts.h"
#include "../src/lib/vision/drwnPixelSegCRFInference.h"
#include "../src/lib/vision/drwnPixelSegModel.h"
#include "../src/lib/vision/drwnSegImageInstance.h"
#include "../src/lib/vision/drwnSegImagePixelFeatures.h"
#include "../src/lib/vision/drwnSegImageRegionFeatures.h"
#include "../src/lib/vision/drwnSuperpixelContainer.h"
#include "../src/lib/vision/drwnTemplateMatcher.h"
#include "../src/lib/vision/drwnTextonFilterBank.h"
#include "../src/lib/vision/drwnVisionUtils.h"

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMultiSegVis.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnMultiSegVis.h
** \anchor drwnMultiSegVis
** \brief Visualization routines for multi-class image segmentation.
** \sa drwnSegImageInstance
** \sa drwnMultiSegRegionDefinitions
*/

#pragma once

#include <cstdlib>
#include <cassert>

#include "drwnBase.h"
#include "drwnIO.h"

#include "drwnMultiSegConfig.h"
#include "drwnSegImageInstance.h"

using namespace std;

// visualization rountines
namespace drwnMultiSegVis {
    //! visualize a labeled drwnSegImageInstance
    cv::Mat visualizeInstance(const drwnSegImageInstance &instance);
    //! overlay pixel labels onto \p canvas (used drwnMultiSegRegionDefinitions)
    void visualizePixelLabels(const drwnSegImageInstance &instance, cv::Mat &canvas,
        double alpha = 0.5);
    //! visualize pixel feature vectors
    cv::Mat visualizePixelFeatures(const drwnSegImageInstance &instance);
};


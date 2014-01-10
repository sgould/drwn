/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImageRegionFeatures.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "cv.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;

// drwnSegImageRegionFeatures class ----------------------------------------

void drwnSegImageRegionFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
    DRWN_LOG_DEBUG("caching data for " << instance.name());
    _instanceName = instance.name();
    _instanceRegions = instance.superpixels.size();
}

void drwnSegImageRegionFeatures::clearInstanceData()
{
    DRWN_LOG_DEBUG("clearing data for " << _instanceName);
    _instanceName.clear();
    _instanceRegions = 0;
}

// drwnSegImageStdRegionFeatures static member variables ------------------

double drwnSegImageStdRegionFeatures::FILTER_BANDWIDTH = 1.0;

// drwnSegImageStdRegionFeatures class -------------------------------------

drwnSegImageStdRegionFeatures::drwnSegImageStdRegionFeatures(const drwnSegImageStdRegionFeatures& f) :
    drwnSegImageRegionFeatures(f), _filters(f._filters)
{
    // do nothing
}

int drwnSegImageStdRegionFeatures::numFeatures() const
{
    return 2 * drwnTextonFilterBank::NUM_FILTERS;
}

void drwnSegImageStdRegionFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
    // cache data needed by parent class
    drwnSegImageRegionFeatures::cacheInstanceData(instance);
    DRWN_FCN_TIC;

    // clear existing filterbank responses
    _filters.clear();

    // filterbank response images
    drwnTextonFilterBank filterBank(FILTER_BANDWIDTH);
    vector<cv::Mat> responses;
    filterBank.filter(instance.image(), responses);
    _filters.addResponseImages(responses);

    DRWN_LOG_DEBUG("...allocated "  << (_filters.memory() / (1024 * 1024))
        << "MB for filter responses");

    DRWN_FCN_TOC;
}

void drwnSegImageStdRegionFeatures::clearInstanceData()
{
    DRWN_LOG_DEBUG("...freeing "  << (_filters.memory() / (1024 * 1024))
        << "MB from filter responses");
    _filters.clear();

    // clear data cached by parent class
    drwnSegImageRegionFeatures::clearInstanceData();
}

void drwnSegImageStdRegionFeatures::appendRegionFeatures(int regId, vector<double>& phi) const
{
    DRWN_ASSERT_MSG(!_filters.empty(), "filter response cache is empty for " << _instanceName);
    DRWN_ASSERT_MSG(regId < _instanceRegions, _instanceName);

    // reserve space for features
    const int nOffset = (int)phi.size();
    phi.reserve(nOffset + this->numFeatures());

    DRWN_TODO;
}

// drwnSegImageRegionFeaturesConfig -----------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnSegImageRegionFeatures
//! \b filterBandwidth :: bandwidth of texton filterbank (default: 1)

class drwnSegImageRegionFeaturesConfig : public drwnConfigurableModule {
public:
    drwnSegImageRegionFeaturesConfig() : drwnConfigurableModule("drwnSegImageRegionFeatures") { }
    ~drwnSegImageRegionFeaturesConfig() { }

    void usage(ostream &os) const {
        os << "      filterBandwidth :: bandwidth of texton filterbank (default: "
           << drwnSegImageStdRegionFeatures::FILTER_BANDWIDTH << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "filterBandwidth")) {
            drwnSegImageStdRegionFeatures::FILTER_BANDWIDTH = std::max(1.0, atof(value));
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnSegImageRegionFeaturesConfig gdrwnSegImageRegionFeaturesConfig;

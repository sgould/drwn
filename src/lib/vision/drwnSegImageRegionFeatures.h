/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImageRegionFeatures.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <list>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;
using namespace Eigen;

// forward declarations ----------------------------------------------------

class drwnSegImageInstance;

// drwnSegImageRegionFeatures class ----------------------------------------
//! Interface for generating per-region (or per-superpixel) features for a
//! drwnSegImageInstance object. The \p superpixel data member of the
//! drwnSegImageInstance object must be populated.
//!
//! \sa drwnSegImagePixelFeatures

class drwnSegImageRegionFeatures : public drwnCloneable {
 protected:
    string _instanceName; //!< name of the chached instance (for error reporting)
    int _instanceRegions; //!< number of superpixels in the cached instance

 public:
    drwnSegImageRegionFeatures() : _instanceRegions(0)
        { /* do nothing */ }
    drwnSegImageRegionFeatures(const drwnSegImageRegionFeatures& f) :
        _instanceName(f._instanceName), _instanceRegions(f._instanceRegions)
        { /* do nothing */ }
    virtual ~drwnSegImageRegionFeatures() { /* do nothing */ }

    drwnSegImageRegionFeatures *clone() const = 0;

    //! return feature vector size
    virtual int numFeatures() const = 0;

    //! caches data for a given drwnSegImageInstance
    virtual void cacheInstanceData(const drwnSegImageInstance& instance);
    //! clears any cached data
    virtual void clearInstanceData();

    //! append features to \p phi for region \p regId in the cached instance
    virtual void appendRegionFeatures(int regId, vector<double>& phi) const = 0;
};

// drwnSegImageStdRegionFeatures class -------------------------------------
//! Standard per-region filterbank features computes mean and standard
//! deviation of drwnTextonFilterBank responses over each region.

class drwnSegImageStdRegionFeatures : public drwnSegImageRegionFeatures {
 public:
    // configurable feature options
    static double FILTER_BANDWIDTH;        //!< bandwidth for filter features

 protected:
    drwnFilterBankResponse _filters;       //!< pixel filter responses

 public:
    drwnSegImageStdRegionFeatures() { /* do nothing */ }
    drwnSegImageStdRegionFeatures(const drwnSegImageStdRegionFeatures& f);
    virtual ~drwnSegImageStdRegionFeatures() { /* do nothing */ }

    drwnSegImageStdRegionFeatures *clone() const {
        return new drwnSegImageStdRegionFeatures(*this);
    }

    // data caching
    void cacheInstanceData(const drwnSegImageInstance& instance);
    void clearInstanceData();

    // feature computation
    int numFeatures() const;
    void appendRegionFeatures(int regId, vector<double>& phi) const;
};

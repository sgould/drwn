/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImagePixelFeatures.h
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

#include "drwnFilterBankResponse.h"

using namespace std;
using namespace Eigen;

// forward declarations ----------------------------------------------------

class drwnSegImageInstance;

// drwnSegImagePixelFeatures class -----------------------------------------
//! Interface for generating per-pixel features for a drwnSegImageInstance
//! object.
//!
//! Derived classes can be used for generating different features, e.g.,
//! filterbank responses, HOG, etc. Typical usage is:
//! \code
//!    f.cacheInstanceData(instance);
//!    f.appendAllPixelFeatures(phi);
//!    f.clearInstanceData();
//! \endcode
//!
//! \sa drwnSegImageRegionFeatures

class drwnSegImagePixelFeatures : public drwnCloneable {
 protected:
    string _instanceName; //!< name of the chached instance (for error reporting)
    int _instanceWidth;   //!< height of the cached instance
    int _instanceHeight;  //!< width of the cached instance

 public:
    drwnSegImagePixelFeatures() : _instanceWidth(0), _instanceHeight(0)
        { /* do nothing */ }
    drwnSegImagePixelFeatures(const drwnSegImagePixelFeatures& pf) :
        _instanceName(pf._instanceName), _instanceWidth(pf._instanceWidth),
        _instanceHeight(pf._instanceHeight) { /* do nothing */ }
    virtual ~drwnSegImagePixelFeatures() { /* do nothing */ }

    drwnSegImagePixelFeatures *clone() const = 0;

    //! return feature vector size
    virtual int numFeatures() const = 0;

    //! caches data for a given drwnSegImageInstance
    virtual void cacheInstanceData(const drwnSegImageInstance& instance);
    //! clears any cached data
    virtual void clearInstanceData();

    //! append features to \p phi for pixel \p (x,y) in the cached instance
    virtual void appendPixelFeatures(int x, int y, vector<double>& phi) const = 0;
    //! append features to \phi for all pixels in the cached instance
    virtual void appendAllPixelFeatures(vector<vector<double> >& phi) const;
};

// drwnSegImageStdPixelFeatures class --------------------------------------
//! Standard per-pixel filterbank features with option to read auxiliary
//! features from a file.

class drwnSegImageStdPixelFeatures : public drwnSegImagePixelFeatures {
 public:
    // configurable feature options
    static double FILTER_BANDWIDTH;        //!< bandwidth for filter features
    static int FEATURE_GRID_SPACING;       //!< cell size for pixel features
    static bool INCLUDE_RGB;               //!< flag to include RGB colour features
    static bool INCLUDE_HOG;               //!< flag to include HOG features
    static bool INCLUDE_LBP;               //!< flag to include LBP features
    static bool INCLUDE_ROWCOLAGG;         //!< flag to include row and column aggregate features
    static bool INCLUDE_LOCATION;          //!< flag to include location features
    static string AUX_FEATURE_DIR;         //!< directory for auxiliary features
    static vector<string> AUX_FEATURE_EXT; //!< auxiliary feature extensions

 protected:
    drwnFilterBankResponse _filters;       //!< pixel filter responses
    vector<vector<double> > _auxFeatures;  //!< cached auxiliary features

 public:
    drwnSegImageStdPixelFeatures() { /* do nothing */ }
    drwnSegImageStdPixelFeatures(const drwnSegImageStdPixelFeatures& pf);
    virtual ~drwnSegImageStdPixelFeatures() { /* do nothing */ }

    drwnSegImageStdPixelFeatures *clone() const {
        return new drwnSegImageStdPixelFeatures(*this);
    }

    // data caching
    void cacheInstanceData(const drwnSegImageInstance& instance);
    void clearInstanceData();

    // feature computation
    int numFeatures() const;
    void appendPixelFeatures(int x, int y, vector<double>& phi) const;
};

// drwnSegImageFilePixelFeatures class -------------------------------------
//! Pre-processed per-pixel features stored in files.

class drwnSegImageFilePixelFeatures : public drwnSegImagePixelFeatures {
 public:
    string featuresDir;          //!< directory containing feature files
    list<string> featuresExt;    //!< feature file extensions

 protected:
    vector<vector<double> > _features;  //!< cached loaded features

 public:
    drwnSegImageFilePixelFeatures() { /* do nothing */ }
    drwnSegImageFilePixelFeatures(const drwnSegImageFilePixelFeatures& pf);
    virtual ~drwnSegImageFilePixelFeatures() { /* do nothing */ }

    drwnSegImageFilePixelFeatures *clone() const {
        return new drwnSegImageFilePixelFeatures(*this);
    }

    // data caching
    void cacheInstanceData(const drwnSegImageInstance& instance);
    void clearInstanceData();

    // feature computation
    int numFeatures() const;
    void appendPixelFeatures(int x, int y, vector<double>& phi) const;
};

// drwnSegImageCompositePixelFeatures class --------------------------------
//! Class for generating composite per-pixel feature vectors.

class drwnSegImageCompositePixelFeatures : public drwnSegImagePixelFeatures {
 protected:
    int _numFeatures;
    std::list<drwnSegImagePixelFeatures *> _featureGenerators;

 public:
    drwnSegImageCompositePixelFeatures() : _numFeatures(0) { /* do nothing */ }
    drwnSegImageCompositePixelFeatures(const drwnSegImageCompositePixelFeatures& pf);
    ~drwnSegImageCompositePixelFeatures();

    drwnSegImageCompositePixelFeatures *clone() const {
        return new drwnSegImageCompositePixelFeatures(*this);
    }

    // data caching
    void cacheInstanceData(const drwnSegImageInstance& instance);
    void clearInstanceData();

    // feature computation
    int numFeatures() const { return _numFeatures; }
    void appendPixelFeatures(int x, int y, vector<double>& phi) const;
    void appendAllPixelFeatures(vector<vector<double> >& phi) const;

    //! clears all feature generators
    void clearFeatureGenerators();
    //! add feature generator (caller gives up ownership)
    void addFeatureGenerator(drwnSegImagePixelFeatures *generator);
    //! copy feature generator (caller must free)
    void copyFeatureGenerator(const drwnSegImagePixelFeatures *generator) {
        DRWN_ASSERT(generator != NULL);
        addFeatureGenerator(generator->clone());
    }
};

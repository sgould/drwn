/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnSegImagePixelFeatures.cpp
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

// drwnSegImagePixelFeatures class -----------------------------------------

void drwnSegImagePixelFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
    DRWN_LOG_DEBUG("caching data for " << instance.name());
    _instanceName = instance.name();
    _instanceWidth = instance.width();
    _instanceHeight = instance.height();
}

void drwnSegImagePixelFeatures::clearInstanceData()
{
    DRWN_LOG_DEBUG("clearing data for " << _instanceName);
    _instanceName.clear();
    _instanceWidth = _instanceHeight = 0;
}

void drwnSegImagePixelFeatures::appendAllPixelFeatures(vector<vector<double> >& phi) const
{
    // check input and sizes
    if (phi.empty()) {
        phi.resize(_instanceWidth * _instanceHeight);
    }
    DRWN_ASSERT(phi.size() == (unsigned)(_instanceWidth * _instanceHeight));

    // append features for each pixel
    int indx = 0;
    for (int y = 0; y < _instanceHeight; y++) {
        for (int x = 0; x < _instanceWidth; x++, indx++) {
            appendPixelFeatures(x, y, phi[indx]);
        }
    }
}

// drwnSegImageStdPixelFeatures static member variables -------------------

double drwnSegImageStdPixelFeatures::FILTER_BANDWIDTH = 1.0;
int drwnSegImageStdPixelFeatures::FEATURE_GRID_SPACING = 5;
bool drwnSegImageStdPixelFeatures::INCLUDE_RGB = false;
bool drwnSegImageStdPixelFeatures::INCLUDE_HOG = false;
bool drwnSegImageStdPixelFeatures::INCLUDE_LBP = false;
bool drwnSegImageStdPixelFeatures::INCLUDE_ROWCOLAGG = false;
bool drwnSegImageStdPixelFeatures::INCLUDE_LOCATION = true;
string drwnSegImageStdPixelFeatures::AUX_FEATURE_DIR;
vector<string> drwnSegImageStdPixelFeatures::AUX_FEATURE_EXT;

// drwnSegImageStdPixelFeatures class --------------------------------------

drwnSegImageStdPixelFeatures::drwnSegImageStdPixelFeatures(const drwnSegImageStdPixelFeatures& pf) :
    drwnSegImagePixelFeatures(pf), _filters(pf._filters), _auxFeatures(pf._auxFeatures)
{
    // do nothing
}

int drwnSegImageStdPixelFeatures::numFeatures() const
{
    int m = drwnTextonFilterBank::NUM_FILTERS;
    if (INCLUDE_RGB) m += 3;
    if (INCLUDE_HOG) m += drwnHOGFeatures::DEFAULT_ORIENTATIONS;
    if (INCLUDE_LBP) m += 4;
    int n = m;
    if (FEATURE_GRID_SPACING > 0) n += 18 * m;
    if (INCLUDE_ROWCOLAGG) n += 4 * m;
    if (INCLUDE_LOCATION) n += 2;
    n += (int)AUX_FEATURE_EXT.size();

    return n;
}

void drwnSegImageStdPixelFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
    // cache data needed by parent class
    drwnSegImagePixelFeatures::cacheInstanceData(instance);
    DRWN_FCN_TIC;

    // clear existing filterbank responses and cached features
    _filters.clear();
    _auxFeatures.clear();

    // filterbank response images
    drwnTextonFilterBank filterBank(FILTER_BANDWIDTH);
    vector<cv::Mat> responses;
    filterBank.filter(instance.image(), responses);
    _filters.addResponseImages(responses);

    // add RGB features
    if (INCLUDE_RGB) {
        cv::Mat rgb(instance.height(), instance.width(), CV_32FC3);
        instance.image().convertTo(rgb, CV_32FC3, 1.0 / 255.0);
        vector<cv::Mat> channels(3);
        cv::split(rgb, &channels[0]);
        _filters.addResponseImages(channels);
    }

    // add HOG features
    if (INCLUDE_HOG) {
        responses.clear();
        drwnHOGFeatures hogFeatureBank;
        hogFeatureBank.computeDenseFeatures(instance.image(), responses);
        _filters.addResponseImages(responses);
    }

    // add LBP features
    if (INCLUDE_LBP) {
        responses.clear();
        drwnLBPFilterBank lbpFilterBank(false);
        lbpFilterBank.filter(instance.image(), responses);
        _filters.addResponseImages(responses);
    }

    // load auxiliary features
    if (!AUX_FEATURE_EXT.empty()) {
        _auxFeatures.resize(instance.size(), vector<double>(AUX_FEATURE_EXT.size()));

        for (unsigned i = 0; i < AUX_FEATURE_EXT.size(); i++) {
            string filename = AUX_FEATURE_DIR + string("/") + _instanceName + AUX_FEATURE_EXT[i];
            DRWN_LOG_DEBUG("...loading features from " << filename);

            string ext(strExtension(AUX_FEATURE_EXT[i]));
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext.compare("txt") == 0) {
                // text file
                ifstream ifs(filename.c_str());
                for (unsigned j = 0; j < _auxFeatures.size(); j++) {
                    ifs >> _auxFeatures[j][i];
                    DRWN_ASSERT_MSG(!ifs.fail(), "expecting " << _auxFeatures.size() <<
                        " features from file text " << filename << " but only read " << j);
                }
                ifs.close();

            } else if (ext.compare("bin") == 0) {
                // 32-bit float
                ifstream ifs(filename.c_str(), ifstream::binary);
                vector<float> buffer(_auxFeatures.size());
                ifs.read((char *)&buffer[0], _auxFeatures.size() * sizeof(float));
                DRWN_ASSERT_MSG(!ifs.fail(), "expecting " << _auxFeatures.size() <<
                    " features from binary file " << filename << " but only read " <<
                    ifs.gcount() / sizeof(float));
                ifs.close();
                for (unsigned j = 0; j < _auxFeatures.size(); j++) {
                    DRWN_ASSERT_MSG(isfinite(buffer[j]), j << "-th feature is not finite");
                    _auxFeatures[j][i] = (double)buffer[j];
                }
            } else {
                DRWN_LOG_FATAL("unknown extension " << AUX_FEATURE_EXT[i]);
            }
        }
    }

    DRWN_LOG_DEBUG("...allocated "  << (_filters.memory() / (1024 * 1024))
        << "MB for filter responses and "
        << (_auxFeatures.size() * AUX_FEATURE_EXT.size() * sizeof(double) / (1024 * 1024))
        << "MB for auxiliary features");

    DRWN_FCN_TOC;
}

void drwnSegImageStdPixelFeatures::clearInstanceData()
{
    DRWN_LOG_DEBUG("...freeing "  << (_filters.memory() / (1024 * 1024))
        << "MB from filter responses and "
        << (_auxFeatures.size() * AUX_FEATURE_EXT.size() * sizeof(double) / (1024 * 1024))
        << "MB for auxiliary features");
    _filters.clear();
    _auxFeatures.clear();

    // clear data cached by parent class
    drwnSegImagePixelFeatures::clearInstanceData();
}

void drwnSegImageStdPixelFeatures::appendPixelFeatures(int x, int y, vector<double>& phi) const
{
    DRWN_ASSERT_MSG(!_filters.empty(), "filter response cache is empty for " << _instanceName);
    DRWN_ASSERT_MSG((x < _instanceWidth) && (y < _instanceHeight), _instanceName);

    // reserve space for features
    const int nOffset = (int)phi.size();
    phi.reserve(nOffset + this->numFeatures());

    // append response at pixel (x,y)
    phi.resize(nOffset + _filters.size());
    Eigen::Map<VectorXd>(&phi[nOffset], _filters.size()) = _filters.value(x, y);

    // append response at various offsets
    if (FEATURE_GRID_SPACING > 0) {
        for (int dy = (int)(-1.5 * FEATURE_GRID_SPACING); dy < FEATURE_GRID_SPACING; dy += FEATURE_GRID_SPACING) {
            for (int dx = (int)(-1.5 * FEATURE_GRID_SPACING); dx < FEATURE_GRID_SPACING; dx += FEATURE_GRID_SPACING) {
                cv::Rect roi(x + dx, y + dy, FEATURE_GRID_SPACING, FEATURE_GRID_SPACING);

                if (roi.x < 0) {
                    roi.width += roi.x;
                    roi.x = 0;
                }
                if (roi.y < 0) {
                    roi.height += roi.y;
                    roi.y = 0;
                }
                roi.width = std::max(std::min(roi.width, _instanceWidth - roi.x), 0);
                roi.height = std::max(std::min(roi.height, _instanceHeight - roi.y), 0);

                if ((roi.width == 0) || (roi.height == 0)) {
                    for (int i = 0; i < _filters.size(); i++) {
                        phi.push_back(phi[nOffset + i]);
                        phi.push_back(phi[nOffset + i]);
                    }
                } else {
                    VectorXd mu = _filters.mean(roi.x, roi.y, roi.width, roi.height);
                    VectorXd var = _filters.variance(roi.x, roi.y, roi.width, roi.height);
                    for (int i = 0; i < _filters.size(); i++) {
                        phi.push_back(mu[i]);
                        phi.push_back(var[i]);
                    }
                }
            }
        }
    }

    // append row and column aggregation features
    if (INCLUDE_ROWCOLAGG) {
        VectorXd muX = _filters.mean(x, 0, 1, _instanceHeight);
        VectorXd muY = _filters.mean(0, y, _instanceWidth, 1);
        VectorXd varX = _filters.variance(x, 0, 1, _instanceHeight);
        VectorXd varY = _filters.variance(0, y, _instanceWidth, 1);
        for (int i = 0; i < _filters.size(); i++) {
            phi.push_back(muX[i]);
            phi.push_back(varX[i]);
            phi.push_back(muY[i]);
            phi.push_back(varY[i]);
        }
    }

    // append row feature (relative height and center line offset)
    if (INCLUDE_LOCATION) {
        phi.push_back((double)y / (double)_instanceHeight);
        phi.push_back(fabs((double)x / (double)_instanceWidth - 0.5));
    }

    // append auxiliary features
    if (!_auxFeatures.empty()) {
        const int indx = y * _instanceWidth + x;
        phi.insert(phi.end(), _auxFeatures[indx].begin(), _auxFeatures[indx].end());
    }
}

// drwnSegImageFilePixelFeatures class -------------------------------------

drwnSegImageFilePixelFeatures::drwnSegImageFilePixelFeatures(const drwnSegImageFilePixelFeatures& pf) :
    drwnSegImagePixelFeatures(pf), featuresDir(pf.featuresDir), featuresExt(pf.featuresExt),
    _features(pf._features)
{
    // do nothing
}

// data caching
void drwnSegImageFilePixelFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
    drwnSegImagePixelFeatures::cacheInstanceData(instance);
    _features.resize(instance.size(), vector<double>(numFeatures(), 0.0));

    // load features from files
    int indx = 0;
    for (list<string>::const_iterator it = featuresExt.begin(); it != featuresExt.end(); ++it) {
        string filename = featuresDir + string("/") + _instanceName + *it;
        DRWN_LOG_VERBOSE("...loading features from " << filename);

        string ext(*it);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext.compare("txt") == 0) {
            // text file
            ifstream ifs(filename.c_str());
            for (unsigned i = 0; i < _features.size(); i++) {
                ifs >> _features[indx][i];
                DRWN_ASSERT_MSG(!ifs.fail(), "file " << filename << " too small");
            }
            ifs.close();

        } else {
            DRWN_LOG_FATAL("unknown extension " << *it);
        }

        indx += 1;
    }
}

void drwnSegImageFilePixelFeatures::clearInstanceData()
{
    _features.clear();
    drwnSegImagePixelFeatures::clearInstanceData();
}

// feature computation
int drwnSegImageFilePixelFeatures::numFeatures() const
{
    return (int)featuresExt.size();
}

void drwnSegImageFilePixelFeatures::appendPixelFeatures(int x, int y, vector<double>& phi) const
{
    const int indx = y * _instanceWidth + x;
    phi.insert(phi.end(), _features[indx].begin(), _features[indx].end());
}

// drwnSegImageCompositePixelFeatures class --------------------------------

drwnSegImageCompositePixelFeatures::drwnSegImageCompositePixelFeatures(const drwnSegImageCompositePixelFeatures& pf) :
    drwnSegImagePixelFeatures(pf)
{
    for (list<drwnSegImagePixelFeatures *>::const_iterator it = pf._featureGenerators.begin();
         it != pf._featureGenerators.end(); ++it) {
        _featureGenerators.push_back((*it)->clone());
    }
}

drwnSegImageCompositePixelFeatures::~drwnSegImageCompositePixelFeatures()
{
    // free memory
    clearFeatureGenerators();
}

void drwnSegImageCompositePixelFeatures::cacheInstanceData(const drwnSegImageInstance& instance)
{
   for (list<drwnSegImagePixelFeatures *>::iterator it = _featureGenerators.begin();
        it != _featureGenerators.end(); ++it) {
       (*it)->cacheInstanceData(instance);
   }
}

void drwnSegImageCompositePixelFeatures::clearInstanceData()
{
   for (list<drwnSegImagePixelFeatures *>::iterator it = _featureGenerators.begin();
        it != _featureGenerators.end(); ++it) {
       (*it)->clearInstanceData();
   }
}

void drwnSegImageCompositePixelFeatures::appendPixelFeatures(int x, int y, vector<double>& phi) const
{
    // append feature vectors from each feature generator
    phi.reserve(phi.size() + _numFeatures);
    for (list<drwnSegImagePixelFeatures *>::const_iterator it = _featureGenerators.begin();
         it != _featureGenerators.end(); ++it) {
        (*it)->appendPixelFeatures(x, y, phi);
    }
}

void drwnSegImageCompositePixelFeatures::appendAllPixelFeatures(vector<vector<double> >& phi) const
{
    // resize phi if no existing features
    phi.resize(_instanceWidth * _instanceHeight);

    // append feature vectors from each feature generator
    for (list<drwnSegImagePixelFeatures *>::const_iterator it = _featureGenerators.begin();
         it != _featureGenerators.end(); ++it) {
        (*it)->appendAllPixelFeatures(phi);
    }
}

void drwnSegImageCompositePixelFeatures::clearFeatureGenerators()
{
    for (list<drwnSegImagePixelFeatures *>::iterator it = _featureGenerators.begin();
         it != _featureGenerators.end(); ++it) {
        delete *it;
    }
    _featureGenerators.clear();
    _numFeatures = 0;
}

void drwnSegImageCompositePixelFeatures::addFeatureGenerator(drwnSegImagePixelFeatures *generator)
{
    DRWN_ASSERT(generator != NULL);
    _featureGenerators.push_back(generator);
    _numFeatures += generator->numFeatures();
}

// drwnSegImagePixelFeaturesConfig ------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnSegImagePixelFeatures
//! \b filterBandwidth :: bandwidth of texton filterbank (default: 1)\n
//! \b featureGridSpacing :: grid spacing for pixel features (default: 5)\n
//! \b includeRGB      :: include RGB colour features (default: false)\n
//! \b includeHOG      :: include dense HOG features (default: false)\n
//! \b includeLBP      :: include LBP features (default: false)\n
//! \b includeRowCol   :: include row and column aggregation (default: false)\n
//! \b includeLocation :: include pixel location in feature vector (default: true)\n
//! \b auxFeatureDir   :: directory for auxiliary features (default: none)\n
//! \b auxFeatureExt   :: space-delimited list of auxiliary feature (default: none)

class drwnSegImagePixelFeaturesConfig : public drwnConfigurableModule {
public:
    drwnSegImagePixelFeaturesConfig() : drwnConfigurableModule("drwnSegImagePixelFeatures") { }
    ~drwnSegImagePixelFeaturesConfig() { }

    void usage(ostream &os) const {
        os << "      filterBandwidth :: bandwidth of texton filterbank (default: "
           << drwnSegImageStdPixelFeatures::FILTER_BANDWIDTH << ")\n";
        os << "      featureGridSpacing :: grid spacing for pixel features (default: "
           << drwnSegImageStdPixelFeatures::FEATURE_GRID_SPACING << ")\n";
        os << "      includeRGB      :: include RGB colour features (default: "
           << (drwnSegImageStdPixelFeatures::INCLUDE_RGB ? "true" : "false") << ")\n";
        os << "      includeHOG      :: include dense HOG features (default: "
           << (drwnSegImageStdPixelFeatures::INCLUDE_HOG ? "true" : "false") << ")\n";
        os << "      includeLBP      :: include LBP features (default: "
           << (drwnSegImageStdPixelFeatures::INCLUDE_LBP ? "true" : "false") << ")\n";
        os << "      includeRowCol   :: include row and column aggregation (default: "
           << (drwnSegImageStdPixelFeatures::INCLUDE_ROWCOLAGG ? "true" : "false") << ")\n";
        os << "      includeLocation :: include pixel location in feature vector (default: "
           << (drwnSegImageStdPixelFeatures::INCLUDE_LOCATION ? "true" : "false") << ")\n";
        os << "      auxFeatureDir   :: directory for auxiliary features (default: "
           << drwnSegImageStdPixelFeatures::AUX_FEATURE_DIR << ")\n";
        os << "      auxFeatureExt   :: space-delimited list of auxiliary feature (default: "
           << toString(drwnSegImageStdPixelFeatures::AUX_FEATURE_EXT) << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "filterBandwidth")) {
            drwnSegImageStdPixelFeatures::FILTER_BANDWIDTH = std::max(1.0, atof(value));
        } else if (!strcmp(name, "featureGridSpacing")) {
            drwnSegImageStdPixelFeatures::FEATURE_GRID_SPACING = std::max(0, atoi(value));
        } else if (!strcmp(name, "includeRGB")) {
            drwnSegImageStdPixelFeatures::INCLUDE_RGB = trueString(string(value));
        } else if (!strcmp(name, "includeHOG")) {
            drwnSegImageStdPixelFeatures::INCLUDE_LBP = trueString(string(value));
        } else if (!strcmp(name, "includeLBP")) {
            drwnSegImageStdPixelFeatures::INCLUDE_HOG = trueString(string(value));
        } else if (!strcmp(name, "includeRowCol")) {
            drwnSegImageStdPixelFeatures::INCLUDE_ROWCOLAGG = trueString(string(value));
        } else if (!strcmp(name, "includeLocation")) {
            drwnSegImageStdPixelFeatures::INCLUDE_LOCATION = trueString(string(value));
        } else if (!strcmp(name, "auxFeatureDir")) {
            drwnSegImageStdPixelFeatures::AUX_FEATURE_DIR = string(value);
        } else if (!strcmp(name, "auxFeatureExt")) {
            drwnSegImageStdPixelFeatures::AUX_FEATURE_EXT.clear();
            parseString<string>(string(value), drwnSegImageStdPixelFeatures::AUX_FEATURE_EXT);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnSegImagePixelFeaturesConfig gdrwnSegImagePixelFeaturesConfig;

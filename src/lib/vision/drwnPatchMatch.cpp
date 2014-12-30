/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPatchMatch.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "cv.h"
#include "highgui.h"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace std;

// drwnPatchMatchEdge --------------------------------------------------------

size_t drwnPatchMatchEdge::numBytesOnDisk() const
{
    return sizeof(float) + sizeof(uint16_t) + sizeof(uint8_t) +
        2 * sizeof(uint16_t) + sizeof(drwnPatchMatchTransform);
}

bool drwnPatchMatchEdge::read(istream& is)
{
    is.read((char *)&matchScore, sizeof(float));
    is.read((char *)&targetNode.imgIndx, sizeof(uint16_t));
    is.read((char *)&targetNode.imgScale, sizeof(uint8_t));
    is.read((char *)&targetNode.xPosition, sizeof(uint16_t));
    is.read((char *)&targetNode.yPosition, sizeof(uint16_t));
    is.read((char *)&xform, sizeof(drwnPatchMatchTransform));
    status = DRWN_PM_DIRTY;

    return true;
}

bool drwnPatchMatchEdge::write(ostream& os) const
{
    os.write((char *)&matchScore, sizeof(float));
    os.write((char *)&targetNode.imgIndx, sizeof(uint16_t));
    os.write((char *)&targetNode.imgScale, sizeof(uint8_t));
    os.write((char *)&targetNode.xPosition, sizeof(uint16_t));
    os.write((char *)&targetNode.yPosition, sizeof(uint16_t));
    os.write((char *)&xform, sizeof(drwnPatchMatchTransform));

    return true;
}

// drwnPatchMatchEdge Utility Functions --------------------------------------

string toString(const drwnPatchMatchEdge& e)
{
    std::stringstream s;
    s << "(" << e.matchScore
      << ", " << e.targetNode.imgIndx
      << "." << e.targetNode.imgScale
      << ", " << e.targetNode.xPosition
      << ", " << e.targetNode.yPosition
      << ", " << (int)e.xform
      << ")";

    if (e.status == DRWN_PM_DIRTY) s << "*";

    return s.str();
}

// drwnPatchMatchImageRecord -------------------------------------------------

void drwnPatchMatchImageRecord::clear()
{
    _matches.clear();
    _matches.resize(_width * _height);
}

bool drwnPatchMatchImageRecord::update(int indx, const drwnPatchMatchEdge& match)
{
    // don't update if new match does not improve the score
    if (match.matchScore >= _matches[indx].back().matchScore)
        return false;

    // search for existing match to the same target image id
    for (drwnPatchMatchEdgeList::iterator it = _matches[indx].begin(); it != _matches[indx].end(); ++it) {
        if (it->targetNode.imgIndx == match.targetNode.imgIndx) {
            if (it->matchScore > match.matchScore) {
                _matches[indx].erase(it);
                drwnPatchMatchEdgeList tmp(1, match);
                _matches[indx].merge(tmp, drwnPatchMatchSortByScore);
                return true;
            } else {
                return false;
            }
        }
    }

    // otherwise remove the previous worst match and add this one
    _matches[indx].pop_back();
    drwnPatchMatchEdgeList tmp(1, match);
    _matches[indx].merge(tmp, drwnPatchMatchSortByScore);

    return true;
}

// drwnPatchMatchImagePyramid ------------------------------------------------

unsigned drwnPatchMatchImagePyramid::MAX_LEVELS = 8;
unsigned drwnPatchMatchImagePyramid::MAX_SIZE = 1024;
unsigned drwnPatchMatchImagePyramid::MIN_SIZE = 32;
double drwnPatchMatchImagePyramid::PYR_SCALE = M_SQRT1_2;

void drwnPatchMatchImagePyramid::clear()
{
    for (size_t i = 0; i < _levels.size(); i++) {
        _levels[i].clear();
    }
}

size_t drwnPatchMatchImagePyramid::numBytesOnDisk() const
{
    size_t metaSize = sizeof(uint16_t) + _name.size() * sizeof(char) +
        2 * sizeof(uint16_t) + sizeof(int);
    size_t dataSize = sizeof(uint16_t);
    for (size_t i = 0; i < _levels.size(); i++) {
        dataSize += 2 * sizeof(uint16_t) + _levels[i].size() * sizeof(uint16_t);
        for (size_t j = 0; j < _levels[i].size(); j++) {
            for (drwnPatchMatchEdgeList::const_iterator kt = (_levels[i])[j].begin(); kt != (_levels[i])[j].end(); ++kt) {
                dataSize += kt->numBytesOnDisk();
            }
        }
    }

    return metaSize + dataSize;
}

bool drwnPatchMatchImagePyramid::write(ostream& os) const
{
    // write meta-data
    uint16_t n = _name.size();
    os.write((char *)&n, sizeof(uint16_t));
    os.write(_name.c_str(), n * sizeof(char));
    os.write((char *)&_width, sizeof(uint16_t));
    os.write((char *)&_height, sizeof(uint16_t));
    n = _levels.size();
    os.write((char *)&n, sizeof(uint16_t));
    os.write((char *)&eqvClass, sizeof(int));

    // write match data
    for (size_t i = 0; i < _levels.size(); i++) {
        n = _levels[i].width();
        os.write((char *)&n, sizeof(uint16_t));
        n = _levels[i].height();
        os.write((char *)&n, sizeof(uint16_t));
        for (size_t j = 0; j < _levels[i].size(); j++) {
            n = (_levels[i])[j].size(); // warning: O(n)
            os.write((char *)&n, sizeof(uint16_t));
            for (drwnPatchMatchEdgeList::const_iterator kt = (_levels[i])[j].begin(); kt != (_levels[i])[j].end(); ++kt) {
                kt->write(os);
            }
        }
    }

    return true;
}

bool drwnPatchMatchImagePyramid::read(istream& is)
{
    // read meta-data
    uint16_t n, m;
    is.read((char *)&n, sizeof(uint16_t));
    char *name = new char[n + 1];
    is.read(name, n * sizeof(char));
    name[n] = '\0';
    _name = string(name);
    delete[] name;
    is.read((char *)&_width, sizeof(uint16_t));
    is.read((char *)&_height, sizeof(uint16_t));
    is.read((char *)&n, sizeof(uint16_t));
    is.read((char *)&eqvClass, sizeof(int));

    DRWN_LOG_DEBUG("loaded " << _name << " of size " << _width << "-by-" << _height << " and "
        << n << " pyramid levels (in equivalence class " << eqvClass << ")");

    // read match data
    _levels.resize(n);
    for (size_t i = 0; i < _levels.size(); i++) {
        is.read((char *)&n, sizeof(uint16_t));
        is.read((char *)&m, sizeof(uint16_t));
        _levels[i] = drwnPatchMatchImageRecord(n, m);
        for (size_t j = 0; j < _levels[i].size(); j++) {
            is.read((char *)&n, sizeof(uint16_t));
            (_levels[i])[j].resize(n);
            for (drwnPatchMatchEdgeList::iterator kt = (_levels[i])[j].begin(); kt != (_levels[i])[j].end(); ++kt) {
                kt->read(is);
            }
        }
    }

    return true;
}

bool drwnPatchMatchImagePyramid::samplePatchByScore(drwnPatchMatchNode& sample) const
{
    float weight = 0.0f;
    for (size_t i = 0; i < _levels.size(); i++) {
        for (size_t j = 0; j < _levels[i].size(); j++) {
            if (!_levels[i][j].empty()) {
                weight += _levels[i][j].front().matchScore;
            }
        }
    }

    weight *= drand48();

    for (size_t i = 0; i < _levels.size(); i++) {
        for (size_t j = 0; j < _levels[i].size(); j++) {
            if (!_levels[i][j].empty()) {
                weight -= _levels[i][j].front().matchScore;
                if (weight <= 0.0) {
                    sample.imgScale = i;
                    sample.xPosition = _levels[i].x(j);
                    sample.yPosition = _levels[i].y(j);
                    return true;
                }
            }
        }
    }

    return false;
}

void drwnPatchMatchImagePyramid::constructPyramidLevels()
{
    DRWN_ASSERT_MSG((PYR_SCALE < 1.0) && (PYR_SCALE > 0.0), PYR_SCALE);
    DRWN_ASSERT_MSG(MAX_SIZE > MIN_SIZE, MAX_SIZE << " >! " << MIN_SIZE);
    _levels.clear();

    // generate pyramid levels
    double scale = std::min(1.0, std::min((double)MAX_SIZE / _width, (double)MAX_SIZE / _height));
    for (unsigned i = 0; i < MAX_LEVELS; i++) {
        const uint16_t w = (uint16_t)(scale * _width);
        const uint16_t h = (uint16_t)(scale * _height);

        DRWN_LOG_DEBUG("adding pyramid level " << i << " (" << w << "-by-" << h << ")");
        if ((w < std::min(MIN_SIZE, drwnPatchMatchGraph::PATCH_WIDTH)) ||
            (h < std::min(MIN_SIZE, drwnPatchMatchGraph::PATCH_HEIGHT)))
            break;

        _levels.push_back(drwnPatchMatchImageRecord(w, h));
        scale *= PYR_SCALE;
    }
}

// drwnPatchMatchGraph -------------------------------------------------------

unsigned int drwnPatchMatchGraph::PATCH_WIDTH = 8;
unsigned int drwnPatchMatchGraph::PATCH_HEIGHT = 8;
unsigned int drwnPatchMatchGraph::K = 10;

drwnPatchMatchGraph::drwnPatchMatchGraph() : imageDirectory(""), imageExtension(""),
    _patchWidth(PATCH_WIDTH), _patchHeight(PATCH_HEIGHT)
{
    // do nothing
}

drwnPatchMatchGraph::drwnPatchMatchGraph(const drwnPatchMatchGraph& graph) :
    imageDirectory(graph.imageDirectory), imageExtension(graph.imageExtension),
    _patchWidth(graph._patchWidth), _patchHeight(graph._patchHeight)
{
    // deep copy images
    _images.reserve(graph._images.size());
    for (unsigned i = 0; i < graph._images.size(); i++) {
        _images.push_back(new drwnPatchMatchImagePyramid(*graph._images[i]));
    }
}

drwnPatchMatchGraph::~drwnPatchMatchGraph()
{
    // free memory
    for (unsigned i = 0; i < _images.size(); i++) {
        delete _images[i];
    }
}

bool drwnPatchMatchGraph::write(const char *filestem) const
{
    DRWN_FCN_TIC;

    // write meta information
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLRootNode(xml, "drwnPatchMatchGraph", false);
    drwnAddXMLAttribute(*node, "dir", imageDirectory.c_str(), false);
    drwnAddXMLAttribute(*node, "ext", imageExtension.c_str(), false);
    drwnAddXMLAttribute(*node, "width", toString(_patchWidth).c_str(), false);
    drwnAddXMLAttribute(*node, "height", toString(_patchHeight).c_str(), false);
    drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);

    for (unsigned i = 0; i < _images.size(); i++) {
        drwnXMLNode *subnode = drwnAddXMLChildNode(*node, "image", NULL, false);
        drwnAddXMLAttribute(*subnode, "index", toString(i).c_str(), false);
        drwnAddXMLAttribute(*subnode, "name", _images[i]->name().c_str(), false);
    }

    ofstream ofs((string(filestem) + string(".meta")).c_str());
    ofs << xml;
    ofs.close();

    // write binary match data
    drwnPersistentStorage storage;
    storage.open(filestem);

#if 1
    storage.clear();
    storage.defragment();
#endif

    for (unsigned i = 0; i < _images.size(); i++) {
        storage.write(_images[i]->name().c_str(), _images[i]);
    }
    storage.close();

    DRWN_FCN_TOC;
    return true;
}

bool drwnPatchMatchGraph::read(const char *filestem)
{
    DRWN_FCN_TIC;

    // read meta information
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, (string(filestem) + string(".meta")).c_str(), "drwnPatchMatchGraph");
    DRWN_ASSERT(node != NULL);

    if (drwnGetXMLAttribute(*node, "dir") != NULL) {
        imageDirectory = string(drwnGetXMLAttribute(*node, "dir"));
    } else {
        imageDirectory.clear();
    }
    if (drwnGetXMLAttribute(*node, "ext") != NULL) {
        imageExtension = string(drwnGetXMLAttribute(*node, "ext"));
    } else {
        imageExtension.clear();
    }
    DRWN_ASSERT(drwnGetXMLAttribute(*node, "width") != NULL);
    _patchWidth = atoi(drwnGetXMLAttribute(*node, "width"));
    DRWN_ASSERT(drwnGetXMLAttribute(*node, "height") != NULL);
    _patchHeight = atoi(drwnGetXMLAttribute(*node, "height"));

    vector<string> baseNames;
    for (drwnXMLNode *subnode = node->first_node("image"); subnode != NULL; subnode = subnode->next_sibling("image")) {
        baseNames.push_back(string(drwnGetXMLAttribute(*subnode, "name")));
    }

    // read binary match data
    drwnPersistentStorage storage;
    storage.open(filestem);

    for (unsigned i = 0; i < _images.size(); i++) {
        delete _images[i];
    }
    _images.clear();
    _images.reserve(baseNames.size());
    for (unsigned i = 0; i < baseNames.size(); i++) {
        _images.push_back(new drwnPatchMatchImagePyramid());
        storage.read(baseNames[i].c_str(), _images.back());
    }
    storage.close();

    DRWN_FCN_TOC;
    return true;
}

int drwnPatchMatchGraph::findImage(const string& baseName) const
{
    for (size_t indx = 0; indx < _images.size(); indx++) {
        if (_images[indx]->name() == baseName) {
            return (int)indx;
        }
    }

    return -1;
}

void drwnPatchMatchGraph::appendImage(const string& baseName)
{
    const string filename = imageFilename(baseName);
    cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, "could not load image " << filename);
    appendImage(baseName, cv::Size(img.cols, img.rows));
}

void drwnPatchMatchGraph::appendImage(const string& baseName, const cv::Size& imgSize)
{
    // ensure that the baseName is not already in the graph
    if (findImage(baseName) != -1) {
        DRWN_LOG_ERROR(baseName << " already exists in the graph");
        return;
    }

    DRWN_LOG_DEBUG("...appending image " << baseName << " (" << toString(imgSize) << ")");
    _images.push_back(new drwnPatchMatchImagePyramid(baseName, imgSize.width, imgSize.height));
}

void drwnPatchMatchGraph::appendImages(const vector<string>& baseNames)
{
    for (unsigned i = 0; i < baseNames.size(); i++) {
        appendImage(baseNames[i]);
    }
}

int drwnPatchMatchGraph::removeImage(unsigned imgIndx)
{
    DRWN_ASSERT(imgIndx < _images.size());

    // remove the image and shift all remaining images down
    delete _images[imgIndx];
    for (unsigned i = imgIndx + 1; i < _images.size(); i++) {
        _images[i - 1] = _images[i];
    }
    _images.resize(_images.size() - 1);

    // update patch references is all edges
    int nEdgesRemoved = 0;
    for (unsigned i = 0; i < _images.size(); i++) {
        for (unsigned j = 0; j < _images[i]->levels(); j++) {
            for (unsigned k = 0; k < (*_images[i])[j].size(); k++) {
                drwnPatchMatchEdgeList &e = (*_images[i])[j][k];
                drwnPatchMatchEdgeList::iterator kt = e.begin();
                while (kt != e.end()) {
                    if (kt->targetNode.imgIndx == imgIndx) {
                        // remove the edge
                        nEdgesRemoved += 1;
                        kt = e.erase(kt);
                    } else {
                        if (kt->targetNode.imgIndx > imgIndx) {
                            // update the image reference
                            kt->targetNode.imgIndx -= 1;
                        }
                        ++kt;
                    }
                }
            }
        }
    }

    return nEdgesRemoved;
}

pair<double, double> drwnPatchMatchGraph::energy() const
{
    DRWN_FCN_TIC;
    double allMatcheEnergy = 0.0;
    double bestMatchEnergy = 0.0;

    for (unsigned imgIndx = 0; imgIndx < _images.size(); imgIndx++) {
        for (unsigned lvlIndx = 0; lvlIndx < _images[imgIndx]->levels(); lvlIndx++) {
            for (unsigned pixIndx = 0; pixIndx < (*_images[imgIndx])[lvlIndx].size(); pixIndx++) {
                const drwnPatchMatchEdgeList& e = (*_images[imgIndx])[lvlIndx][pixIndx];
                if (e.empty()) continue;
                bestMatchEnergy += e.front().matchScore;
                for (drwnPatchMatchEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                    allMatcheEnergy += kt->matchScore;
                }
            }
        }
    }

    DRWN_FCN_TOC;
    return make_pair(allMatcheEnergy, bestMatchEnergy);
}

// drwnPatchMatchGraphLearner ------------------------------------------------

double drwnPatchMatchGraphLearner::SEARCH_DECAY_RATE = 0.5;
int drwnPatchMatchGraphLearner::FORWARD_ENRICHMENT_K = 3;
bool drwnPatchMatchGraphLearner::DO_INVERSE_ENRICHMENT = true;
bool drwnPatchMatchGraphLearner::DO_LOCAL_SEARCH = true;
bool drwnPatchMatchGraphLearner::DO_EXHAUSTIVE = false;

drwnPatchMatchTransform drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS = DRWN_PM_TRANSFORM_NONE;
double drwnPatchMatchGraphLearner::TOP_VAR_PATCHES = 1.0;

#ifdef DRWN_DEBUG_STATISTICS
unsigned drwnPatchMatchGraphLearner::_dbPatchesScored = 0;
#endif

drwnPatchMatchGraphLearner::drwnPatchMatchGraphLearner(drwnPatchMatchGraph& graph) :
    _graph(graph)
{
    // cache features
    cacheImageFeatures();
}

drwnPatchMatchGraphLearner::~drwnPatchMatchGraphLearner()
{
#ifdef DRWN_DEBUG_STATISTICS
    // display debug statistics
    if (_dbPatchesScored > 0) {
        DRWN_LOG_MESSAGE("drwnPatchMatchGraphLearner computed " << _dbPatchesScored
            << " patch comparisons");
    }
#endif
}

void drwnPatchMatchGraphLearner::initialize()
{
    DRWN_ASSERT_MSG(_graph.size() > drwnPatchMatchGraph::K,
        "must have more images than matched per pixel");

    //! \todo cache image features here instead of in constructor?

    // initialize matches
    for (unsigned imgIndx = 0; imgIndx < _graph.size(); imgIndx++) {
        if (!_graph[imgIndx].bActive) {
            DRWN_LOG_DEBUG("...skipping initialization of " << _graph[imgIndx].name());
            continue;
        }

        // initialize the image
        initialize(imgIndx);
    }
}

void drwnPatchMatchGraphLearner::initialize(unsigned imgIndx)
{
    DRWN_FCN_TIC;
    DRWN_LOG_DEBUG("...initializing " << _graph[imgIndx].name());

    // initialize indicies for fast subsampling
    vector<unsigned> indexes;
    indexes.reserve(_graph.size() - 1);
    for (unsigned i = 0; i < _graph.size(); i++) {
        if (i == imgIndx) continue;
        if ((_graph[imgIndx].eqvClass >= 0) && (_graph[imgIndx].eqvClass == _graph[i].eqvClass))
            continue;
        indexes.push_back(i);
    }
    if (indexes.size() < drwnPatchMatchGraph::K) {
        DRWN_LOG_WARNING(_graph[imgIndx].name() << " has less than " <<
            drwnPatchMatchGraph::K << " images to match against");
    }

    // load image so can sort pixel locations by patch variance
    cv::Mat img = cv::imread(_graph.imageFilename(imgIndx), CV_LOAD_IMAGE_COLOR);

    // initialize all pixel locations at all pyramid levels
    for (unsigned lvlIndx = 0; lvlIndx < _graph[imgIndx].levels(); lvlIndx++) {

        // resize image to this pyramid level and sort pixel locations by variance
        drwnResizeInPlace(img, _graph[imgIndx][lvlIndx].height(), _graph[imgIndx][lvlIndx].width());
        vector<cv::Point> pixels = drwnPatchMatchUtils::sortPixelsByVariance(img,
            cv::Size(_graph.patchWidth(), _graph.patchHeight()));

        for (unsigned p = 0; p < pixels.size(); p++) {
            // determine number of matches to add
            const unsigned kPixel = std::min((unsigned)indexes.size(), (pixels.size() < 1024) ||
                (p < TOP_VAR_PATCHES * pixels.size()) ? drwnPatchMatchGraph::K : 1);

            const drwnPatchMatchNode u(imgIndx, lvlIndx, pixels[p].x, pixels[p].y);
            drwnPatchMatchEdgeList& e = _graph.edges(u);
            bool bHasExistingMatches = !e.empty();

            // randomly initialize K matches (to K different images)
            vector<unsigned> indxB = drwn::subSample(indexes, kPixel);
            for (unsigned k = 0; k < kPixel; k++) {

                const uint8_t lvlB = (uint8_t)(drand48() * _graph[indxB[k]].levels());
                const cv::Point topLeftB(int(drand48() * (_graph[indxB[k]][lvlB].width() - _graph.patchWidth())),
                    int(drand48() * (_graph[indxB[k]][lvlB].height() - _graph.patchHeight())));
                const drwnPatchMatchNode v(indxB[k], lvlB, topLeftB);

                drwnPatchMatchTransform patchXform = DRWN_PM_TRANSFORM_NONE;
                if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00) {
                    if (drand48() < 0.5) patchXform |= DRWN_PM_TRANSFORM_HFLIP;
                }
                if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_VFLIP) != 0x00) {
                    if (drand48() < 0.5) patchXform |= DRWN_PM_TRANSFORM_VFLIP;
                }

                const float score = scoreMatch(u, v, patchXform);
                DRWN_ASSERT(score != DRWN_FLT_MAX);
                e.push_back(drwnPatchMatchEdge(score, v, patchXform));
            }

            // remove excess matches if existing
            if (bHasExistingMatches) {
                // remove duplicate target images
                e.sort(drwnPatchMatchSortByImage);
                drwnPatchMatchEdgeList::iterator kt = e.begin();
                drwnPatchMatchEdgeList::iterator jt = kt++;
                while (kt != e.end()) {
                    if (kt->targetNode.imgIndx == jt->targetNode.imgIndx) {
                        kt = e.erase(kt);
                    } else {
                        jt = kt++;
                    }
                }
            }

            // sort matches from best to worst
            e.sort(drwnPatchMatchSortByScore);

            // resize to correct number of matches per pixel
            if (bHasExistingMatches) {
                e.resize(kPixel);
            }
        }
    }

    DRWN_FCN_TOC;
}

void drwnPatchMatchGraphLearner::rescore()
{
    DRWN_FCN_TIC;

    for (unsigned imgIndx = 0; imgIndx < _graph.size(); imgIndx++) {
        for (unsigned imgScale = 0; imgScale < _graph[imgIndx].levels(); imgScale++) {
            for (unsigned pixIndx = 0; pixIndx < _graph[imgIndx][imgScale].size(); pixIndx++) {

                const drwnPatchMatchNode u(imgIndx, imgScale, _graph[imgIndx][imgScale].index2pixel(pixIndx));
                drwnPatchMatchEdgeList& e = _graph[imgIndx][imgScale][pixIndx];

                // rescore matches
                for (drwnPatchMatchEdgeList::iterator kt = e.begin(); kt != e.end(); ++kt) {
                    kt->matchScore = scoreMatch(u, kt->targetNode, kt->xform);
                }

                // sort matches from best to worst
                e.sort(drwnPatchMatchSortByScore);
            }
        }
    }

    DRWN_FCN_TOC;
}

void drwnPatchMatchGraphLearner::update()
{
    DRWN_FCN_TIC;

    // try improve a bad match (from a random (active) image)
    if (DO_EXHAUSTIVE) {
        vector<unsigned> activeIndx;
        activeIndx.reserve(_graph.size());
        for (unsigned imgIndx = 0; imgIndx < _graph.size(); imgIndx++) {
            if (_graph[imgIndx].bActive)
                activeIndx.push_back(imgIndx);
        }

        drwnPatchMatchNode u;
        u.imgIndx = activeIndx[(unsigned)(drand48() * activeIndx.size())];
        if (_graph[u.imgIndx].samplePatchByScore(u)) {
            exhaustive(u);
        }
    }

    // progagation and decaying search
    for (unsigned imgIndx = 0; imgIndx < _graph.size(); imgIndx++) {
        // check if image is in active set
        if (!_graph[imgIndx].bActive) {
            DRWN_LOG_DEBUG("...skipping " << _graph[imgIndx].name());
            continue;
        }

        // perform update
        //! \todo this can be parallelized
        update(imgIndx);
    }

    // enrichment (forward and inverse)
    enrichment();

    DRWN_FCN_TOC;
}

void drwnPatchMatchGraphLearner::update(unsigned imgIndx)
{
    DRWN_LOG_DEBUG("...updating " << _graph[imgIndx].name());

    drwnPatchMatchNode u(imgIndx, 0, 0, 0);
    for (unsigned imgScale = 0; imgScale < _graph[imgIndx].levels(); imgScale++) {
        u.imgScale = imgScale;

        // forward neighbourhood propagation, local search and decaying search
        for (unsigned pixIndx = 0; pixIndx < _graph[imgIndx][imgScale].size(); pixIndx++) {
            u.yPosition = _graph[imgIndx][imgScale].y(pixIndx);
            u.xPosition = _graph[imgIndx][imgScale].x(pixIndx);

            if (DO_LOCAL_SEARCH) {
                local(u);
            }
            propagate(u, true);
            search(u);
        }

        // backward neighbourhood propagation and local search
        for (unsigned pixIndx = _graph[imgIndx][imgScale].size(); pixIndx > 0; pixIndx--) {
            u.yPosition = _graph[imgIndx][imgScale].y(pixIndx - 1);
            u.xPosition = _graph[imgIndx][imgScale].x(pixIndx - 1);

            if (DO_LOCAL_SEARCH) {
                local(u);
            }
            propagate(u, false);
        }
    }
}

bool drwnPatchMatchGraphLearner::propagate(const drwnPatchMatchNode& u, bool bDirection)
{
    drwnPatchMatchEdgeList& e = _graph.edges(u);
    if (e.empty()) return false;

    bool bChanged = false;
    for (drwnPatchMatchEdgeList::iterator kt = e.begin(); kt != e.end(); ++kt) {

        // check if match has already been processed in a previous iteration
        if (kt->status == DRWN_PM_DIRTY) {
            kt->status = DRWN_PM_PASS1;
        } else if (kt->status == DRWN_PM_PASS1) {
            kt->status = DRWN_PM_PASS2;
        } else continue;

        // target node
        const drwnPatchMatchNode& v = kt->targetNode;

        if (!bDirection) {
            if (u.yPosition > 0) {
                // find node pair to the north of current edge
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale, u.xPosition, u.yPosition - 1);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition,
                    (kt->xform & DRWN_PM_TRANSFORM_VFLIP) == 0x00 ? std::max(v.yPosition - 1, 0) :
                    std::min(v.yPosition + 1, (int)(_graph[v.imgIndx][v.imgScale].height() - _graph.patchHeight())));
                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }

            if (u.xPosition > 0) {
                // find node pair to the west of current edge
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale, u.xPosition - 1, u.yPosition);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale,
                    (kt->xform & DRWN_PM_TRANSFORM_HFLIP) == 0x00 ? std::max(v.xPosition - 1, 0) :
                    std::min(v.xPosition + 1, (int)(_graph[v.imgIndx][v.imgScale].width() - _graph.patchWidth())), v.yPosition);
                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }

            if ((u.imgScale > 0) && (v.imgScale > 0)) {
                // find node pair in pyramid level one up from current edge
                const cv::Point pu = mapPatch(cv::Point(u.xPosition, u.yPosition), u.imgIndx,
                    u.imgScale, u.imgScale - 1);
                const cv::Point pv = mapPatch(cv::Point(v.xPosition, v.yPosition), v.imgIndx,
                    v.imgScale, v.imgScale - 1);
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale - 1, pu.x, pu.y);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale - 1, pv.x, pv.y);
                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }

        } else {
            if (u.yPosition < int(_graph[u.imgIndx][u.imgScale].height() - _graph.patchHeight())) {
                // find node pair to the south of current edge
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale, u.xPosition, u.yPosition + 1);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition,
                    (kt->xform & DRWN_PM_TRANSFORM_VFLIP) != 0x00 ? std::max(v.yPosition - 1, 0) :
                    std::min(v.yPosition + 1, (int)(_graph[v.imgIndx][v.imgScale].height() - _graph.patchHeight())));

                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }

            if (u.xPosition < int(_graph[u.imgIndx][u.imgScale].width() - _graph.patchWidth())) {
                // find node pair to the north of the current edge
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale, u.xPosition + 1, u.yPosition);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale,
                    (kt->xform & DRWN_PM_TRANSFORM_HFLIP) != 0x00 ? std::max(v.xPosition - 1, 0) :
                    std::min(v.xPosition + 1, (int)(_graph[v.imgIndx][v.imgScale].width() - _graph.patchWidth())), v.yPosition);

                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }

            if (((unsigned)u.imgScale + 1 < _graph[u.imgIndx].levels()) &&
                ((unsigned)v.imgScale + 1 < _graph[v.imgIndx].levels())) {
                // find node pair in pyramid level one down from current edge
                const cv::Point pu = mapPatch(cv::Point(u.xPosition, u.yPosition), u.imgIndx,
                    u.imgScale, u.imgScale + 1);
                const cv::Point pv = mapPatch(cv::Point(v.xPosition, v.yPosition), v.imgIndx,
                    v.imgScale, v.imgScale + 1);
                const drwnPatchMatchNode uu(u.imgIndx, u.imgScale + 1, pu.x, pu.y);
                const drwnPatchMatchNode vv(v.imgIndx, v.imgScale + 1, pv.x, pv.y);
                drwnPatchMatchEdgeList& s = _graph[uu.imgIndx][uu.imgScale](uu.xPosition, uu.yPosition);
                if (!s.empty()) {
                    const float score = scoreMatch(uu, vv, kt->xform, s.back().matchScore);
                    const drwnPatchMatchEdge match(score, vv, kt->xform);
                    if (_graph[uu.imgIndx][uu.imgScale].update(uu.xPosition, uu.yPosition, match)) {
                        bChanged = true;
                    }
                }
            }
        }
    }

    return bChanged;
}

bool drwnPatchMatchGraphLearner::search(const drwnPatchMatchNode& u)
{
    // skip search step if decay rate is out of range
    if ((SEARCH_DECAY_RATE <= 0.0) || (SEARCH_DECAY_RATE >= 1.0))
        return false;

    bool bChanged = false;
    drwnPatchMatchEdgeList& e = _graph.edges(u);

    for (drwnPatchMatchEdgeList::iterator kt = e.begin(); kt != e.end(); ++kt) {

        const int searchLimit = (kt->status == DRWN_PM_PASS2) ? 2 : 1;
        const drwnPatchMatchNode &v = kt->targetNode;

        const int imgWidth = _graph[v.imgIndx][v.imgScale].width();
        const int imgHeight = _graph[v.imgIndx][v.imgScale].height();

        double alpha = 1.0;
        while (1) {
            // search range
            const int alphaWidth = (int)(alpha * imgWidth);
            const int alphaHeight = (int)(alpha * imgHeight);
            const int xmin = std::max(0, v.xPosition - alphaWidth);
            const int xmax = std::min((int)(imgWidth - _graph.patchWidth()),
                v.xPosition + alphaWidth);
            const int ymin = std::max(0, v.yPosition - alphaHeight);
            const int ymax = std::min((int)(imgHeight - _graph.patchHeight()),
                v.yPosition + alphaHeight);

            if ((xmax - xmin < searchLimit) || (ymax - ymin < searchLimit))
                break;

            // sample scale
            const int smin = std::max((int)v.imgScale - 1, 0);
            const int smax = std::min((int)v.imgScale + 1, (int)_graph[v.imgIndx].levels());
            const int imgScale = (int)(smin + drand48() * (smax - smin));

            // sample patch
            const cv::Point p = mapPatch(cv::Point((int)(xmin + drand48() * (xmax - xmin)),
                    (int)(ymin + drand48() * (ymax - ymin))), v.imgIndx, v.imgScale, imgScale);
            const drwnPatchMatchNode vv(v.imgIndx, imgScale, p.x, p.y);

            // sample transform
            drwnPatchMatchTransform xform = DRWN_PM_TRANSFORM_NONE;
            if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00) {
                if (drand48() < 0.5) xform |= DRWN_PM_TRANSFORM_HFLIP;
            }
            if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_VFLIP) != 0x00) {
                if (drand48() < 0.5) xform |= DRWN_PM_TRANSFORM_VFLIP;
            }

            // score sampled patch
            const float score = scoreMatch(u, vv, xform, kt->matchScore);

            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode = vv;
                kt->xform = xform;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
                break;
            }

            alpha *= SEARCH_DECAY_RATE;
        }
    }

    // sort matches from best to worst
    if (bChanged) {
        e.sort(drwnPatchMatchSortByScore);
    }

    return bChanged;
}

bool drwnPatchMatchGraphLearner::local(const drwnPatchMatchNode& u)
{
    bool bChanged = false;
    drwnPatchMatchEdgeList& e = _graph.edges(u);

    for (drwnPatchMatchEdgeList::iterator kt = e.begin(); kt != e.end(); ++kt) {
        if (kt->status == DRWN_PM_PASS2) continue;

        drwnPatchMatchNode v(kt->targetNode);

        // search y direction
        if (v.yPosition > 0) {
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition, v.yPosition - 1);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.yPosition = vv.yPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }

        if (v.yPosition < _graph[v.imgIndx][v.imgScale].height() - _graph.patchHeight()) {
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition, v.yPosition + 1);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.yPosition = vv.yPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }

        v.yPosition = kt->targetNode.yPosition;

        // search x direction
        if (v.xPosition > 0) {
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition - 1, v.yPosition);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.xPosition = vv.xPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }

        if (v.xPosition < _graph[v.imgIndx][v.imgScale].width() - _graph.patchWidth()) {
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale, v.xPosition + 1, v.yPosition);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.xPosition = vv.xPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }

        v.xPosition = kt->targetNode.xPosition;

        // search pyramid level up
        if (v.imgScale > 0) {
            const cv::Point p = mapPatch(cv::Point(v.xPosition, v.yPosition), v.imgIndx,
                v.imgScale, v.imgScale - 1);
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale - 1, p.x, p.y);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.imgScale = vv.imgScale;
                kt->targetNode.xPosition = vv.xPosition;
                kt->targetNode.yPosition = vv.yPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }

        // search pyramid level down
        if ((unsigned)v.imgScale + 1 < _graph[v.imgIndx].levels()) {
            const cv::Point p = mapPatch(cv::Point(v.xPosition, v.yPosition), v.imgIndx,
                v.imgScale, v.imgScale + 1);
            const drwnPatchMatchNode vv(v.imgIndx, v.imgScale + 1, p.x, p.y);
            const float score = scoreMatch(u, vv, kt->xform, kt->matchScore);
            if (score < kt->matchScore) {
                kt->matchScore = score;
                kt->targetNode.imgScale = vv.imgScale;
                kt->targetNode.xPosition = vv.xPosition;
                kt->targetNode.yPosition = vv.yPosition;
                kt->status = DRWN_PM_DIRTY;
                bChanged = true;
            }
        }
    }

    // sort matches from best to worst
    if (bChanged) {
        e.sort(drwnPatchMatchSortByScore);
    }

    return bChanged;
}

bool drwnPatchMatchGraphLearner::enrichment()
{
    bool bChanged = false;

    const int hEnrichment = drwnCodeProfiler::getHandle("drwnPatchMatchGraphLearner::enrichment");
    for (unsigned imgIndx = 0; imgIndx < _graph.size(); imgIndx++) {
        DRWN_LOG_DEBUG("...enrichment step for " << _graph[imgIndx].name());
        drwnCodeProfiler::tic(hEnrichment);

        // inverse enrichment
        if (DO_INVERSE_ENRICHMENT) {
            for (unsigned imgScale = 0; imgScale < _graph[imgIndx].levels(); imgScale++) {
                for (unsigned pixIndx = 0; pixIndx < _graph[imgIndx][imgScale].size(); pixIndx++) {
                    // node matching to
                    const drwnPatchMatchNode v(imgIndx, imgScale, _graph[imgIndx][imgScale].index2pixel(pixIndx));
                    const drwnPatchMatchEdgeList& e = _graph[imgIndx][imgScale][pixIndx];

                    for (drwnPatchMatchEdgeList::const_iterator kt = e.begin(); kt != e.end(); ++kt) {
                        const drwnPatchMatchNode &u = kt->targetNode;
                        if (!_graph[u.imgIndx].bActive)
                            continue;

                        drwnPatchMatchEdgeList& s = _graph.edges(u);
                        if (s.empty()) continue; // pixels are not being processed

                        const drwnPatchMatchEdge match(kt->matchScore, v, kt->xform);

                        if (_graph[u.imgIndx][u.imgScale].update(u.xPosition, u.yPosition, match)) {
                            bChanged = true;
                        }
                    }
                }
            }
        } // end: inverse enrichment

        // forward enrichment
        if ((FORWARD_ENRICHMENT_K > 0) && _graph[imgIndx].bActive) {

            for (unsigned imgScale = 0; imgScale < _graph[imgIndx].levels(); imgScale++) {
                for (unsigned pixIndx = 0; pixIndx < _graph[imgIndx][imgScale].size(); pixIndx++) {
                    // node matching from
                    const drwnPatchMatchNode u(imgIndx, imgScale, _graph[imgIndx][imgScale].index2pixel(pixIndx));

                    // needed to prevent invalid iterators when updating e
                    const drwnPatchMatchEdgeList r(_graph[imgIndx][imgScale][pixIndx]);

                    int outerLoopCounter = 0;
                    for (drwnPatchMatchEdgeList::const_iterator it = r.begin(); it != r.end(); ++it) {
                        if (++outerLoopCounter > FORWARD_ENRICHMENT_K)
                            break;

                        const drwnPatchMatchEdgeList& s = _graph.edges(it->targetNode);

                        int innerLoopCounter = 0;
                        for (drwnPatchMatchEdgeList::const_iterator kt = s.begin(); kt != s.end(); ++kt) {

                            // check that we're not matching back to ourself (or any other
                            // image in the same equivalence class)
                            if (kt->targetNode.imgIndx == imgIndx) continue;
                            if ((_graph[imgIndx].eqvClass >= 0) &&
                                (_graph[imgIndx].eqvClass == _graph[kt->targetNode.imgIndx].eqvClass))
                                continue;

                            if ((it->status == DRWN_PM_PASS2) &&
                                (kt->status == DRWN_PM_PASS2)) continue;

                            if (++innerLoopCounter > FORWARD_ENRICHMENT_K)
                                break;

                            const drwnPatchMatchTransform xform = drwnComposeTransforms(kt->xform, it->xform);
                            const float score = scoreMatch(u, kt->targetNode, xform, _graph.edges(u).back().matchScore);

                            const drwnPatchMatchEdge match(score, kt->targetNode, xform);
                            if (_graph[u.imgIndx][u.imgScale].update(u.xPosition, u.yPosition, match)) {
                                bChanged = true;
                            }
                        }
                    }
                }
            }
        } // end: forward enrichment

        drwnCodeProfiler::toc(hEnrichment);
    }

    return bChanged;
}

bool drwnPatchMatchGraphLearner::exhaustive(const drwnPatchMatchNode& u)
{
    DRWN_FCN_TIC;

    // prepare set of allowed transforms
    set<drwnPatchMatchTransform> allowedTransforms;
    allowedTransforms.insert(DRWN_PM_TRANSFORM_NONE);
    if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00) {
        allowedTransforms.insert(DRWN_PM_TRANSFORM_HFLIP);
    }
    if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_VFLIP) != 0x00) {
        allowedTransforms.insert(DRWN_PM_TRANSFORM_VFLIP);
        if ((ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00) {
            allowedTransforms.insert(DRWN_PM_TRANSFORM_VFLIP | DRWN_PM_TRANSFORM_HFLIP);
        }
    }

    // get list of edges
    drwnPatchMatchEdgeList& e = _graph.edges(u);

    bool bChanged = false;
    drwnPatchMatchNode v;
    for (v.imgIndx = 0; v.imgIndx < _graph.size(); v.imgIndx++) {
        // skip if same image or in same equivalence class
        if (v.imgIndx == u.imgIndx) continue;
        if ((_graph[u.imgIndx].eqvClass >= 0) && (_graph[u.imgIndx].eqvClass == _graph[v.imgIndx].eqvClass))
            continue;

        for (v.imgScale = 0; v.imgScale < _graph[v.imgIndx].levels(); v.imgScale++) {

            // find best scoring match for this image (that is at least as good as
            // the worst match found so far, e.back())
            drwnPatchMatchEdge bestMatch;
            if (!e.empty()) bestMatch = e.back();

            // search over transforms
            for (set<drwnPatchMatchTransform>::const_iterator it = allowedTransforms.begin(); it != allowedTransforms.end(); ++it) {
                // search over locations
                for (v.yPosition = 0; v.yPosition < _graph[v.imgIndx][v.imgScale].height() -
                         _graph.patchHeight() + 1; v.yPosition++) {
                    for (v.xPosition = 0; v.xPosition < _graph[v.imgIndx][v.imgScale].width() -
                             _graph.patchWidth() + 1; v.xPosition++) {

                        const float score = scoreMatch(u, v, *it, bestMatch.matchScore);
                        if (score < bestMatch.matchScore) {
                            bestMatch = drwnPatchMatchEdge(score, v, *it);
                        }
                    }
                }
            }

            DRWN_ASSERT(bestMatch.matchScore != DRWN_FLT_MAX);
            if (_graph[u.imgIndx][u.imgScale].update(u.xPosition, u.yPosition, bestMatch)) {
                bChanged = true;
            }
        }
    }

    DRWN_FCN_TOC;
    return bChanged;
}

void drwnPatchMatchGraphLearner::cacheImageFeatures()
{
    DRWN_ASSERT(_features.empty());
    _features.resize(_graph.size());
    for (unsigned i = 0; i < _graph.size(); i++) {
        cacheImageFeatures(i);
    }
}

void drwnPatchMatchGraphLearner::cacheImageFeatures(unsigned imgIndx)
{
    DRWN_FCN_TIC;

    // load image
    cv::Mat img = cv::imread(_graph.imageFilename(imgIndx), CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(img.data != NULL, _graph.imageFilename(imgIndx) << " [" << imgIndx << "]");

    // compute features for each pyramid level
    _features[imgIndx].resize(_graph[imgIndx].levels());
    for (unsigned lvlIndx = 0; lvlIndx < _graph[imgIndx].levels(); lvlIndx++) {

        // resize image for this pyramid level
        drwnResizeInPlace(img, _graph[imgIndx][lvlIndx].height(), _graph[imgIndx][lvlIndx].width());

        // create feature image
        _features[imgIndx][lvlIndx] = cv::Mat(img.rows, img.cols, CV_8UC(5));
        int nChannel = appendCIELabFeatures(img, _features[imgIndx][lvlIndx], 0);
        nChannel = appendEdgeFeatures(img, _features[imgIndx][lvlIndx], nChannel);
        nChannel = appendVerticalFeatures(img, _features[imgIndx][lvlIndx], nChannel);
        DRWN_ASSERT(nChannel == _features[imgIndx][lvlIndx].channels());
    }

    DRWN_FCN_TOC;
}

int drwnPatchMatchGraphLearner::appendCIELabFeatures(const cv::Mat& img, cv::Mat& features, int nChannel) const
{
    DRWN_ASSERT(nChannel + 3 <= features.channels());

    cv::Mat lab(img.rows, img.cols, CV_8UC3);
    cv::cvtColor(img, lab, CV_BGR2Lab);

    int from_to[] = {0, 0};
    for (int c = 0; c < 3; c++) {
        from_to[0] = c;
        from_to[1] = c + nChannel;
        cv::mixChannels(&lab, 1, &features, 1, from_to, 1);
    }

    return nChannel + 3;
}

int drwnPatchMatchGraphLearner::appendVerticalFeatures(const cv::Mat& img, cv::Mat& features, int nChannel) const
{
    DRWN_ASSERT(nChannel < features.channels());
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            features.at<unsigned char>(y, features.channels() * x + nChannel) =
                (unsigned char)(255 * y / img.rows);
        }
    }
    nChannel += 1;

    return nChannel;
}

int drwnPatchMatchGraphLearner::appendEdgeFeatures(const cv::Mat& img, cv::Mat& features, int nChannel) const
{
    DRWN_ASSERT(nChannel < features.channels());

    cv::Mat edges = drwnSoftEdgeMap(img, true);
    cv::Mat edgesU8(img.rows, img.cols, CV_8UC1);
    edges.convertTo(edgesU8, CV_8U, 255.0, 0.0);

    int from_to[] = {0, nChannel};
    cv::mixChannels(&edgesU8, 1, &features, 1, from_to, 1);

    nChannel += 1;
    return nChannel;
}

#define DISTANCE_METRIC(X) abs(X)
//#define DISTANCE_METRIC(X) ((X) * (X))

float drwnPatchMatchGraphLearner::scoreMatch(const drwnPatchMatchNode& u,
    const drwnPatchMatchNode& v, const drwnPatchMatchTransform& xform, float maxValue) const
{
    DRWN_ASSERT_MSG(u.imgIndx != v.imgIndx, u.imgIndx << " == " << v.imgIndx);
    DRWN_ASSERT(_features[u.imgIndx][u.imgScale].data != NULL);
    DRWN_ASSERT(_features[v.imgIndx][v.imgScale].data != NULL);

#ifdef DRWN_DEBUG_STATISTICS
    _dbPatchesScored += 1;
#endif

    // score match (sum-of-absolute-differences, SAD) or (sum-of-squares difference, SSD)
    float score = 0.0;

    const int nFeatures = _features[v.imgIndx][v.imgScale].channels();
    const int nWidth = (int)_graph.patchWidth();
    const int nHeight = (int)_graph.patchHeight();

    //DRWN_ASSERT(_features[u.imgIndx][u.imgScale].step[1] == nFeatures);
    //DRWN_ASSERT(_features[v.imgIndx][v.imgScale].step[1] == nFeatures);

#if 0
    // slow approach (for debugging)
    for (int y = 0; y < nHeight; y++) {
        for (int x = 0; x < nWidth; x++) {
            for (int c = 0; c < nFeatures; c++) {
                const float f_u = _features[u.imgIndx][u.imgScale].at<unsigned char>(u.yPosition + y, 
                    (u.xPosition + x) * nFeatures + c);
                const int x_v = (xform & DRWN_PM_TRANSFORM_HFLIP) == 0x00 ? 
                    v.xPosition + x : v.xPosition + nWidth - x - 1;
                const int y_v = (xform & DRWN_PM_TRANSFORM_VFLIP) == 0x00 ? 
                    v.yPosition + y : v.yPosition + nHeight - y - 1;
                const float f_v = _features[v.imgIndx][v.imgScale].at<unsigned char>(y_v, x_v * nFeatures + c);
                DRWN_ASSERT_MSG(isfinite(f_u) && isfinite(f_v), f_u << " " << f_v 
                    << " (" << x << ", " << y << ", " << c << ")"
                    << " (" << (u.xPosition + x) << ", " << (u.yPosition + y) << ", " << c << ")"
                    << " (" << x_v << ", " << y_v << ", " << c << ")"); 
                score += DISTANCE_METRIC(f_u - f_v);
            }
        }
        if (score > maxValue) break;
    }

    return score;
#endif

    const ptrdiff_t qColStep = (xform & DRWN_PM_TRANSFORM_HFLIP) == 0x00 ? nFeatures : -nFeatures;
    const ptrdiff_t qRowStep = ((xform & DRWN_PM_TRANSFORM_VFLIP) == 0x00 ?
        _features[v.imgIndx][v.imgScale].step[0] :
        0 - _features[v.imgIndx][v.imgScale].step[0]) - qColStep * nWidth;
    const ptrdiff_t pRowStep = _features[u.imgIndx][u.imgScale].step[0] - nWidth * nFeatures;

    const unsigned char *p = _features[u.imgIndx][u.imgScale].ptr<unsigned char>(u.yPosition) + nFeatures * u.xPosition;
    const unsigned char *q = _features[v.imgIndx][v.imgScale].ptr<unsigned char>(
        (xform & DRWN_PM_TRANSFORM_VFLIP) == 0x00 ? v.yPosition : v.yPosition + nHeight - 1) +
        ((xform & DRWN_PM_TRANSFORM_HFLIP) == 0x00 ? nFeatures * v.xPosition : nFeatures * (v.xPosition + nWidth - 1));

    switch (nFeatures) {
    case 1:
      for (int y = 0; y < nHeight; y++) {
            unsigned iSum = 0;
            for (int x = 0; x < nWidth; ++x, p += 1, q += qColStep) {
                iSum += DISTANCE_METRIC(*p - *q);
            }

            score += (float)iSum;
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
        break;

    case 2:
      for (int y = 0; y < nHeight; y++) {

            unsigned iSum = 0;
            for (int x = 0; x < nWidth; ++x, p += 2, q += qColStep) {
                iSum += DISTANCE_METRIC(p[0] - q[0]) + DISTANCE_METRIC(p[1] - q[1]);
            }

            score += (float)iSum;
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
        break;

    case 3:
        for (int y = 0; y < nHeight; y++) {

            unsigned rSum = 0;
            unsigned gSum = 0;
            unsigned bSum = 0;
            for (int x = 0; x < nWidth; x++, p += 3, q += qColStep) {
                rSum += DISTANCE_METRIC(p[0] - q[0]);
                gSum += DISTANCE_METRIC(p[1] - q[1]);
                bSum += DISTANCE_METRIC(p[2] - q[2]);
            }

            score += (float)(rSum + gSum + bSum);
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
        break;

    case 4:
        for (int y = 0; y < nHeight; y++) {

            unsigned iSum = 0;
            for (int x = 0; x < nWidth; x++, p += 4, q += qColStep) {
                iSum += DISTANCE_METRIC(p[0] - q[0]) + DISTANCE_METRIC(p[1] - q[1]) +
                    DISTANCE_METRIC(p[2] - q[2]) + DISTANCE_METRIC(p[3] - q[3]);
            }

            score += (float)iSum;
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
        break;

    case 5:
        for (int y = 0; y < nHeight; y++) {

            unsigned iSum = 0;
            for (int x = 0; x < nWidth; x++, p += 5, q += qColStep) {
                iSum += DISTANCE_METRIC(p[0] - q[0]) + DISTANCE_METRIC(p[1] - q[1]) +
                    DISTANCE_METRIC(p[2] - q[2]) + DISTANCE_METRIC(p[3] - q[3]) +
                    DISTANCE_METRIC(p[4] - q[4]);
            }

            score += (float)iSum;
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
        break;

    default:
        for (int y = 0; y < nHeight; y++) {

            unsigned iSum = 0;
            for (int x = 0; x < nWidth; ++x, p += nFeatures, q += qColStep) {
                for (int c = 0; c < nFeatures; c++) {
                    iSum += DISTANCE_METRIC(p[c] - q[c]);
                }
            }

            score += (float)iSum;
            if (score > maxValue) break;

            p += pRowStep;
            q += qRowStep;
        }
    }

    return score;
}

cv::Point drwnPatchMatchGraphLearner::mapPatch(const cv::Point& p, int imgIndx,
    int srcScale, int dstScale) const
{
    // check if scales are the same
    if (srcScale == dstScale) return p;

    cv::Point q = _graph[imgIndx].mapPixel(cv::Point(p.x + _graph.patchWidth() / 2,
            p.y + _graph.patchHeight() / 2), srcScale, dstScale);

    q.x = std::max(q.x - (int)_graph.patchWidth() / 2, 0);
    q.y = std::max(q.y - (int)_graph.patchHeight() / 2, 0);

    // check that we haven't overstepped the image boundary
    if (q.x + _graph.patchWidth() > _graph[imgIndx][dstScale].width()) {
        q.x = _graph[imgIndx][dstScale].width() - _graph.patchWidth();
    }
    if (q.y + _graph.patchHeight() > _graph[imgIndx][dstScale].height()) {
        q.y = _graph[imgIndx][dstScale].height() - _graph.patchHeight();
    }

    return q;
}

// drwnPatchMatchRetarget ----------------------------------------------------

drwnPatchMatchGraphRetarget::drwnPatchMatchGraphRetarget(const drwnPatchMatchGraph& graph) :
    _graph(graph)
{
    // cache labels
    cacheImageLabels();
}

drwnPatchMatchGraphRetarget::~drwnPatchMatchGraphRetarget()
{
    // do nothing
}

cv::Mat drwnPatchMatchGraphRetarget::retarget(unsigned imgIndx) const
{
    DRWN_ASSERT(imgIndx < _graph.size());

    // retargetted canvas
    cv::Mat canvas = cv::Mat::zeros(_graph[imgIndx].height(), _graph[imgIndx].width(), CV_8UC3);

    // patch score used for retargetting
    cv::Mat score(canvas.rows, canvas.cols, CV_32FC1, cv::Scalar(DRWN_FLT_MAX));

    for (unsigned imgScale = 0; imgScale < _graph[imgIndx].levels(); imgScale++) {

        // pyramid scaling factor
        const double yScale = (double)canvas.rows / (double)_graph[imgIndx][imgScale].height();
        const double xScale = (double)canvas.cols / (double)_graph[imgIndx][imgScale].width();

        const int nHeight = (int)(yScale * _graph.patchHeight());
        const int nWidth = (int)(xScale * _graph.patchWidth());
        cv::Mat patch(nHeight, nWidth, CV_8UC3);

        for (int y = 0; y < _graph[imgIndx][imgScale].height(); y++) {
            for (int x = 0; x < _graph[imgIndx][imgScale].width(); x++) {
                const drwnPatchMatchEdgeList& e = _graph.edges(drwnPatchMatchNode(imgIndx, imgScale, x, y));
                if (e.empty()) continue;

                const cv::Rect dstRect((int)(xScale * x), (int)(yScale * y), nWidth, nHeight);

                const drwnPatchMatchNode& r = e.front().targetNode;
                const double srcYScale = (double)_labels[r.imgIndx].rows /
                    (double)_graph[r.imgIndx][r.imgScale].height();
                const double srcXScale = (double)_labels[r.imgIndx].cols /
                    (double)_graph[r.imgIndx][r.imgScale].width();
                const cv::Rect srcRect((int)(srcXScale * r.xPosition), (int)(srcYScale * r.yPosition),
                    (int)(srcXScale * _graph.patchWidth()), (int)(srcYScale * _graph.patchHeight()));

                cv::resize(_labels[r.imgIndx](srcRect), patch, cv::Size(nHeight, nWidth), CV_INTER_LINEAR);

                for (int v = 0; v < dstRect.height; v++) {
                    for (int u = 0; u < dstRect.width; u++) {
                        if (e.front().matchScore < score.at<float>(dstRect.y + v, dstRect.x + u)) {
                            canvas.at<cv::Vec3b>(dstRect.y + v, dstRect.x + u) = patch.at<cv::Vec3b>(v, u);
                            score.at<float>(dstRect.y + v, dstRect.x + u) = e.front().matchScore;
                        }
                    }
                }
            }
        }
    }

    return canvas;
}

void drwnPatchMatchGraphRetarget::cacheImageLabels()
{
    DRWN_ASSERT(_labels.empty());
    _labels.resize(_graph.size());
    for (unsigned i = 0; i < _graph.size(); i++) {
        cacheImageLabels(i);
    }
}

void drwnPatchMatchGraphRetarget::cacheImageLabels(unsigned imgIndx)
{
    // load image
    _labels[imgIndx] = cv::imread(_graph.imageFilename(imgIndx), CV_LOAD_IMAGE_COLOR);
    DRWN_ASSERT_MSG(_labels[imgIndx].data != NULL, _graph.imageFilename(imgIndx));
}

// drwnPatchMatchVisualization -----------------------------------------------

cv::Mat drwnPatchMatchVis::visualizeMatches(const drwnPatchMatchGraph &graph,
    int imgIndx, const cv::Point& p)
{
    vector<cv::Mat> views;

    views.push_back(cv::imread(graph.imageFilename(imgIndx), CV_LOAD_IMAGE_COLOR));
    drwnResizeInPlace(views.back(), graph[imgIndx][0].height(), graph[imgIndx][0].width());
    drwnDrawBoundingBox(views.back(), cv::Rect(p.x, p.y, graph.patchWidth(), graph.patchHeight()),
        CV_RGB(255, 0, 0));

    // only show matches for base scale
    const drwnPatchMatchEdgeList& e = graph[imgIndx][0](p.x, p.y);
    for (drwnPatchMatchEdgeList::const_iterator it = e.begin(); it != e.end(); ++it) {
        views.push_back(cv::imread(graph.imageFilename(it->targetNode.imgIndx), CV_LOAD_IMAGE_COLOR));
        drwnResizeInPlace(views.back(), graph[it->targetNode.imgIndx][0].height(),
            graph[it->targetNode.imgIndx][0].width());
        drwnDrawBoundingBox(views.back(), cv::Rect(it->targetNode.xPosition, it->targetNode.yPosition,
                graph.patchWidth(), graph.patchHeight()), CV_RGB(0, 0, 255));
        drwnResizeInPlace(views.back(), views[0].rows, views[0].cols);
    }

    // combine into a single canvas
    return drwnCombineImages(views);
}

cv::Mat drwnPatchMatchVis::visualizeMatchQuality(const drwnPatchMatchGraph &graph,
    int imgIndx, float maxScore)
{
    // view quality for the first pyramid level
    cv::Mat quality = cv::Mat::zeros(graph[imgIndx][0].height(), graph[imgIndx][0].width(), CV_32FC1);

    for (int y = 0; y < quality.rows; y++) {
        for (int x = 0; x < quality.cols; x++) {
            const drwnPatchMatchEdgeList& e = graph[imgIndx][0](x, y);
            if (!e.empty()) {
                maxScore = std::max(maxScore, e.front().matchScore);
                quality.at<float>(y, x) = e.front().matchScore;
            }
        }
    }

    // scale to [0, 1]
    quality /= maxScore;

    // convert to heatmap
    return drwnCreateHeatMap(quality, DRWN_COLORMAP_RAINBOW);
}

cv::Mat drwnPatchMatchVis::visualizeMatchQuality(const drwnPatchMatchGraph &graph)
{
    // determine max score
    float maxScore = 0.0f;
    for (unsigned i = 0; i < graph.size(); i++) {
        if (!graph[i].bActive) continue;
        for (unsigned p = 0; p < graph[i][0].size(); p++) {
            if (!graph[i][0][p].empty()) {
                maxScore = std::max(maxScore, graph[i][0][p].front().matchScore);
            }
        }
    }

    vector<cv::Mat> views;
    for (int imgIndx = 0; imgIndx < (int)graph.size(); imgIndx++) {
        if (!graph[imgIndx].bActive) continue;
        views.push_back(drwnPatchMatchVis::visualizeMatchQuality(graph, imgIndx, maxScore));
        drwnResizeInPlace(views.back(), views[0].rows, views[0].cols);
    }

    return drwnCombineImages(views);
}

cv::Mat drwnPatchMatchVis::visualizeMatchTransforms(const drwnPatchMatchGraph &graph,
    int imgIndx)
{
    cv::Mat canvas = cv::Mat::zeros(graph[imgIndx][0].height(), graph[imgIndx][0].width(), CV_8UC3);

    for (int y = 0; y < canvas.rows; y++) {
        for (int x = 0; x < canvas.cols; x++) {
            const drwnPatchMatchEdgeList& e = graph[imgIndx][0](x, y);
            if (!e.empty()) {
                if ((e.front().xform & DRWN_PM_TRANSFORM_HFLIP) != 0x00) {
                    canvas.at<unsigned char>(y, 3 * x + 2) = 0xff;
                }
                if ((e.front().xform & DRWN_PM_TRANSFORM_VFLIP) != 0x00) {
                    canvas.at<unsigned char>(y, 3 * x + 1) = 0xff;
                }
            }
        }
    }

    return canvas;
}

cv::Mat drwnPatchMatchVis::visualizeMatchTransforms(const drwnPatchMatchGraph &graph)
{
    vector<cv::Mat> views;
    for (int imgIndx = 0; imgIndx < (int)graph.size(); imgIndx++) {
        if (!graph[imgIndx].bActive) continue;
        views.push_back(drwnPatchMatchVis::visualizeMatchTransforms(graph, imgIndx));
        drwnResizeInPlace(views.back(), views[0].rows, views[0].cols);
    }

    return drwnCombineImages(views);
}

cv::Mat drwnPatchMatchVis::visualizeMatchTargets(const drwnPatchMatchGraph &graph, int imgIndx)
{
    cv::Mat m = cv::Mat::zeros(graph[imgIndx][0].height(), graph[imgIndx][0].width(), CV_32FC1);

    for (int y = 0; y < m.rows; y++) {
        for (int x = 0; x < m.cols; x++) {
            const drwnPatchMatchEdgeList& e = graph[imgIndx][0](x, y);
            if (!e.empty()) {
                m.at<float>(y, x) = (float)e.front().targetNode.imgIndx + 1.0f;
            }
        }
    }

    m /= (float)graph.size();
    return drwnCreateHeatMap(m);
}

cv::Mat drwnPatchMatchVis::visualizeMatchTargets(const drwnPatchMatchGraph &graph)
{
    vector<cv::Mat> views;
    for (int imgIndx = 0; imgIndx < (int)graph.size(); imgIndx++) {
        if (!graph[imgIndx].bActive) continue;
        views.push_back(drwnPatchMatchVis::visualizeMatchTargets(graph, imgIndx));
        drwnResizeInPlace(views.back(), views[0].rows, views[0].cols);
    }

    return drwnCombineImages(views);
}

// drwnPatchMatchConfig -----------------------------------------------------
//! \addtogroup drwnConfigSettings
//! \section drwnPatchMatch
//! \b maxPyramidLevels :: maximum pyramid levels (default: 8)\n
//! \b maxImageSize    :: maximum image dimension in pyramid (default: 1024)\n
//! \b minImageSize    :: minimum image dimension in pyramid (default: 32)\n
//! \b pyramidScale    :: downsample rate for image pyramid (default: 0.71)\n
//! \b patchWidth      :: patch width at base scale (default: 8)\n
//! \b patchHeight     :: patch height at base scale (default: 8)\n
//! \b K               :: matches per pixel (default: 10)\n
//! \b decayRate       :: exponential search decay rate (default: 0.5)\n
//! \b fwdEnrichment   :: forward enrichment search depth (default: 3)\n
//! \b invEnrichment   :: inverse enrichment (default: yes)\n
//! \b localSearch     :: neighbourhood search on dirty pixels (default: yes)\n
//! \b randomExhaustive :: exhaustive search on a random pixel (default: no)\n
//! \b topVarPatches    :: initialize highest variance matches to K and others to 1 (default: 1.0)\n
//! \b allowHFlips     :: allow patches to be flipped horizontally during search (default: no)\n
//! \b allowVFlips     :: allow patches to be flipped vertically during search (default: no)\n

class drwnPatchMatchConfig : public drwnConfigurableModule {
public:
    drwnPatchMatchConfig() : drwnConfigurableModule("drwnPatchMatch") { }
    ~drwnPatchMatchConfig() { }

    void usage(ostream &os) const {

        os << "      maxPyramidLevels :: maximum pyramid levels (default: "
           << drwnPatchMatchImagePyramid::MAX_LEVELS << ")\n";
        os << "      maxImageSize    :: maximum image dimension in pyramid (default: "
           << drwnPatchMatchImagePyramid::MAX_SIZE << ")\n";
        os << "      minImageSize    :: minimum image dimension in pyramid (default: "
           << drwnPatchMatchImagePyramid::MIN_SIZE << ")\n";
        os << "      pyramidScale    :: downsample rate for image pyramid (default: "
           << drwnPatchMatchImagePyramid::PYR_SCALE << ")\n";

        os << "      patchWidth      :: patch width at base scale (default: "
           << drwnPatchMatchGraph::PATCH_WIDTH << ")\n";
        os << "      patchHeight     :: patch height at base scale (default: "
           << drwnPatchMatchGraph::PATCH_HEIGHT << ")\n";
        os << "      K               :: matches per pixel (default: "
           << drwnPatchMatchGraph::K << ")\n";

        os << "      decayRate       :: exponential search decay rate (default: "
           << drwnPatchMatchGraphLearner::SEARCH_DECAY_RATE << ")\n";
        os << "      fwdEnrichment   :: forward enrichment search depth (default: "
           << drwnPatchMatchGraphLearner::FORWARD_ENRICHMENT_K << ")\n";
        os << "      invEnrichment   :: inverse enrichment (default: "
           << (drwnPatchMatchGraphLearner::DO_INVERSE_ENRICHMENT ? "yes" : "no") << ")\n";
        os << "      localSearch     :: neighbourhood search on dirty pixels (default: "
           << (drwnPatchMatchGraphLearner::DO_LOCAL_SEARCH ? "yes" : "no") << ")\n";
        os << "      randomExhaustive :: exhaustive search on a random pixel (default: "
           << (drwnPatchMatchGraphLearner::DO_EXHAUSTIVE ? "yes" : "no") << ")\n";
        os << "      topVarPatches   :: initialize highest variance matches to K and others to 1 (default: "
           << drwnPatchMatchGraphLearner::TOP_VAR_PATCHES << ")\n";
        os << "      allowHFlips     :: allow patches to be flipped horizontally during search (default: "
           << ((drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_HFLIP) != 0x00 ? "yes" : "no") << ")\n";
        os << "      allowVFlips     :: allow patches to be flipped vertically during search (default: "
           << ((drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS & DRWN_PM_TRANSFORM_VFLIP) != 0x00 ? "yes" : "no") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "maxPyramidLevels")) {
            drwnPatchMatchImagePyramid::MAX_LEVELS = std::max(1, atoi(value));
        } else if (!strcmp(name, "maxImageSize")) {
            drwnPatchMatchImagePyramid::MAX_SIZE = std::max(32, atoi(value));
        } else if (!strcmp(name, "minImageSize")) {
            drwnPatchMatchImagePyramid::MIN_SIZE = std::max(1, atoi(value));
        } else if (!strcmp(name, "pyramidScale")) {
            drwnPatchMatchImagePyramid::PYR_SCALE = std::max(0.0, atof(value));
        } else if (!strcmp(name, "patchWidth")) {
            drwnPatchMatchGraph::PATCH_WIDTH = std::max(1, atoi(value));
        } else if (!strcmp(name, "patchHeight")) {
            drwnPatchMatchGraph::PATCH_HEIGHT = std::max(1, atoi(value));
        } else if (!strcmp(name, "K")) {
            drwnPatchMatchGraph::K = std::max(1, atoi(value));
        } else if (!strcmp(name, "decayRate")) {
            drwnPatchMatchGraphLearner::SEARCH_DECAY_RATE = std::min(std::max(0.0, atof(value)), 1.0);
        } else if (!strcmp(name, "fwdEnrichment")) {
            drwnPatchMatchGraphLearner::FORWARD_ENRICHMENT_K = atoi(value);
        } else if (!strcmp(name, "invEnrichment")) {
            drwnPatchMatchGraphLearner::DO_INVERSE_ENRICHMENT = drwn::trueString(string(value));
        } else if (!strcmp(name, "localSearch")) {
            drwnPatchMatchGraphLearner::DO_LOCAL_SEARCH = drwn::trueString(string(value));
        } else if (!strcmp(name, "randomExhaustive")) {
            drwnPatchMatchGraphLearner::DO_EXHAUSTIVE = drwn::trueString(string(value));
        } else if (!strcmp(name, "allowHFlips")) {
            if (drwn::trueString(string(value))) {
                drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS |= DRWN_PM_TRANSFORM_HFLIP;
            } else {
                drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS &= ~DRWN_PM_TRANSFORM_HFLIP;
            }
        } else if (!strcmp(name, "allowVFlips")) {
            if (drwn::trueString(string(value))) {
                drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS |= DRWN_PM_TRANSFORM_VFLIP;
            } else {
                drwnPatchMatchGraphLearner::ALLOWED_TRANSFORMATIONS &= ~DRWN_PM_TRANSFORM_VFLIP;
            }
        } else if (!strcmp(name, "topVarPatches")) {
            drwnPatchMatchGraphLearner::TOP_VAR_PATCHES = std::min(std::max(0.0, atof(value)), 1.0);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnPatchMatchConfig gPatchMatchConfig;

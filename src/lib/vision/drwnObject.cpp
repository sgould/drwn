/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnObject.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <vector>
#include <iomanip>

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;

// drwnObject ---------------------------------------------------------------

drwnObject::drwnObject() :
    name(""), extent(0, 0, 0, 0), score(0.0), ref(-1)
{
    // do nothing
}

drwnObject::drwnObject(const cv::Rect& ext, const char *n) :
    name(""), extent(ext), score(0.0), ref(-1)
{
    if (n != NULL) name = string(n);
}

drwnObject::drwnObject(const drwnObject& obj) :
    name(obj.name), extent(obj.extent), score(obj.score), ref(obj.ref)
{
    // do nothing
}

drwnObject::~drwnObject()
{
    // do nothing
}

bool drwnObject::save(drwnXMLNode& xml) const
{
    drwnAddXMLAttribute(xml, "name", name.c_str(), false);
    drwnAddXMLAttribute(xml, "x", toString(extent.x).c_str(), false);
    drwnAddXMLAttribute(xml, "y", toString(extent.y).c_str(), false);
    drwnAddXMLAttribute(xml, "w", toString(extent.width).c_str(), false);
    drwnAddXMLAttribute(xml, "h", toString(extent.height).c_str(), false);
    if (score != 0.0) {
        drwnAddXMLAttribute(xml, "score", toString(score).c_str(), false);
    }
    return true;
}

bool drwnObject::load(drwnXMLNode& xml)
{
    name = string(drwnGetXMLAttribute(xml, "name"));
    extent.x = atoi(drwnGetXMLAttribute(xml, "x"));
    extent.y = atoi(drwnGetXMLAttribute(xml, "y"));
    extent.width = atoi(drwnGetXMLAttribute(xml, "w"));
    extent.height = atoi(drwnGetXMLAttribute(xml, "h"));
    if (drwnGetXMLAttribute(xml, "score") == NULL) {
        score = 0.0;
    } else {
        score = atof(drwnGetXMLAttribute(xml, "score"));
    }

    return true;
}

bool drwnObject::hit(double x, double y) const
{
    return ((x >= extent.x) && (x <= extent.x + extent.width) &&
        (y >= extent.y) && (y <= extent.y + extent.height));
}

double drwnObject::overlap(const cv::Rect& roi) const
{
    int iw, ih;

    // find region of overlap
    iw = std::min(extent.x + extent.width, roi.x + roi.width) - std::max(extent.x, roi.x);
    ih = std::min(extent.y + extent.height, roi.y + roi.height) - std::max(extent.y, roi.y);

    if ((iw < 0) || (ih < 0.0)) {
        return 0.0;
    }

    // return area of intersection
    return (double)(iw * ih);
}


drwnObject& drwnObject::operator=(const drwnObject& obj)
{
    if (&obj != this) {
        name = obj.name;
        extent = obj.extent;
        score = obj.score;
        ref = obj.ref;
    }
    return *this;
}

bool drwnObject::operator==(const drwnObject& obj) const
{
    return ((name == obj.name) && (extent == obj.extent) && (score == obj.score));
}

bool drwnObject::operator<(const drwnObject& obj) const
{
    if (score == obj.score) {
        return (area() < obj.area());
    }
    return (score < obj.score);
}

bool drwnObject::operator>(const drwnObject& obj) const
{
    if (score == obj.score) {
        return (area() > obj.area());
    }
    return (score > obj.score);
}

// drwnObjectList -----------------------------------------------------------

drwnObjectList::drwnObjectList()
{
    // do nothing
}

drwnObjectList::~drwnObjectList()
{
    // do nothing
}


bool drwnObjectList::save(drwnXMLNode& xml) const
{
    for (drwnObjectList::const_iterator it = begin(); it != end(); ++it) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnObject", NULL, false);
        it->save(*node);
    }

    return true;
}

bool drwnObjectList::load(drwnXMLNode& xml)
{
    clear();

    const int numObjects = drwnCountXMLChildren(xml, "drwnObject");
    reserve(numObjects);

    for (drwnXMLNode *node = xml.first_node("drwnObject"); node != NULL;
         node = node->next_sibling("drwnObject")) {
        DRWN_ASSERT(!drwnIsXMLEmpty(*node));
        push_back(drwnObject());
        back().load(*node);
    }

    return true;
}

void drwnObjectList::sort()
{
    std::stable_sort(this->begin(), this->end(), std::greater<drwnObject>());
}

int drwnObjectList::removeNonMatching(const string& name)
{
    drwnObjectList filteredFrame;
    for (drwnObjectList::const_iterator it = begin(); it != end(); it++) {
        if (it->name == name) {
            filteredFrame.push_back(*it);
        }
    }

    int nRemoved = (int)(this->size() - filteredFrame.size());
    this->swap(filteredFrame);
    return nRemoved;
}

int drwnObjectList::removeMatching(const string& name)
{
    drwnObjectList filteredFrame;
    for (drwnObjectList::const_iterator it = begin(); it != end(); it++) {
        if (it->name != name) {
            filteredFrame.push_back(*it);
        }
    }

    int nRemoved = (int)(this->size() - filteredFrame.size());
    this->swap(filteredFrame);
    return nRemoved;
}

int drwnObjectList::nonMaximalSuppression(double threshold, bool bByName)
{
    DRWN_FCN_TIC;
    if (size() < 2) {
        DRWN_FCN_TOC;
        return 0;
    }

    // sort objects by decreasing score
    sort();

    // decide which objects to include in the output
    const drwnObjectList& frame = *this;
    vector<bool> includeInOutput(frame.size(), true);
    vector<double> areas(frame.size());
    for (unsigned i = 0; i < frame.size(); i++) {
        areas[i] = frame[i].area();
    }

    for (unsigned i = 0; i < frame.size(); i++) {
        if (!includeInOutput[i]) continue;
        for (unsigned j = i + 1; j < frame.size(); j++) {
            if (!includeInOutput[j]) continue;
            if (bByName && (frame[i].name != frame[j].name)) {
                continue;
            }

            double ao = frame[i].overlap(frame[j]);
#if 0
            if (ao > threshold * (areas[i] + areas[j] - ao)) {
                includeInOutput[j] = false;
            }
#else
            if (ao > threshold * areas[j]) {
                includeInOutput[j] = false;
            }
#endif
        }
    }

    // remove suppressed frames
    unsigned insertIndx = 0;
    for (unsigned i = 0; i < includeInOutput.size(); i++) {
        if (includeInOutput[i]) {
            if (i != insertIndx) {
                (*this)[insertIndx] = (*this)[i];
            }
            insertIndx += 1;
        }
    }

    this->resize(insertIndx);

    DRWN_FCN_TOC;
    return (int)(includeInOutput.size() - this->size());
}

int drwnObjectList::keepHighestScoring(int nKeep)
{
    int nRemoved = std::min((int)size(), (int)size() - nKeep);
    if (nKeep <= 0) {
        clear();
    } else {
        sort();
        if (nKeep < (int)size()) {
            resize(nKeep);
        }
    }

    return nRemoved;
}

int drwnObjectList::keepAboveScore(double threshold)
{
    // sort detections by score (highest first)
    sort();
    unsigned cutoffIndex = 0;
    while (cutoffIndex < this->size()) {
        if ((*this)[cutoffIndex].score < threshold) break;
        cutoffIndex++;
    }

    int nRemoved = (int)(size() - cutoffIndex);
    this->resize(cutoffIndex);
    return nRemoved;
}

// drwnObjectSequence -------------------------------------------------------

drwnObjectSequence::drwnObjectSequence()
{
    // do nothing
}

drwnObjectSequence::~drwnObjectSequence()
{
    // do nothing
}

bool drwnObjectSequence::save(drwnXMLNode& xml) const
{
    for (drwnObjectSequence::const_iterator it = begin(); it != end(); ++it) {
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnObjectList", NULL, false);
        drwnAddXMLAttribute(*node, "id", it->first.c_str(), false);
        it->second.save(*node);
    }

    return true;
}

bool drwnObjectSequence::load(drwnXMLNode& xml)
{
    clear();

    for (drwnXMLNode *node = xml.first_node("drwnObjectList"); node != NULL;
         node = node->next_sibling("drwnObjectList")) {
        DRWN_ASSERT(!drwnIsXMLEmpty(*node) && (node->first_attribute("id") != NULL));
        
        string id = string(node->first_attribute("id")->value());
        if (this->find(id) != this->end()) {
            DRWN_LOG_ERROR("duplicate id \"" << id << "\" in drwnObjectSequence");
            continue;
        }

        drwnObjectSequence::iterator it = this->insert(make_pair(id, drwnObjectList())).first;
        it->second.load(*node);
    }

    return true;
}

// utility functions
void printObjectStatistics(const drwnObjectSequence& objects)
{
    map<string, int> objectIndex;
    vector<int> count;
    vector<cv::Size> minBoundingBox;
    vector<cv::Size> maxBoundingBox;

    // accumulate results
    for (drwnObjectSequence::const_iterator it = objects.begin(); it != objects.end(); ++it) {
        for (drwnObjectList::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt) {
            map<string, int>::iterator kt = objectIndex.find(jt->name);
            if (kt == objectIndex.end()) {
                kt = objectIndex.insert(make_pair(jt->name, (int)count.size())).first;
                count.push_back(0);
                minBoundingBox.push_back(cv::Size(jt->extent.width, jt->extent.height));
                maxBoundingBox.push_back(cv::Size(jt->extent.width, jt->extent.height));
            }

            count[kt->second] += 1;
            if (minBoundingBox[kt->second].width * minBoundingBox[kt->second].height >
                jt->extent.width * jt->extent.height) {
                minBoundingBox[kt->second] = cv::Size(jt->extent.width, jt->extent.height);
            }
            if (maxBoundingBox[kt->second].width * maxBoundingBox[kt->second].height <
                jt->extent.width * jt->extent.height) {
                maxBoundingBox[kt->second] = cv::Size(jt->extent.width, jt->extent.height);
            }
        }
    }

    // display results
    for (map<string, int>::const_iterator it = objectIndex.begin(); it != objectIndex.end(); it++) {
        cout << setw(16) << it->first << "\t"
             << setw(8) << count[it->second] << "\t"
             << minBoundingBox[it->second].width << "-by-" << minBoundingBox[it->second].height << "\t"
             << maxBoundingBox[it->second].width << "-by-" << maxBoundingBox[it->second].height << "\n";
    }
}

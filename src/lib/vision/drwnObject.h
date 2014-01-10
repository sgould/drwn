/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnObject.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>

#include "cv.h"
#include "cxcore.h"

#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;

// drwnObject ---------------------------------------------------------------
//! Encapsulates a 2D object in an image for object detection.

class drwnObject {
 public:
    string name;         //!< name of the object category, e.g., "car"
    cv::Rect extent;     //!< object bounding box (in pixels)
    double score;        //!< score (probability) for the object (higher is better)
    int ref;             //!< external reference

 public:
    //! default constructor
    drwnObject();
    //! construct an object with given extent \p ext and name \p n
    drwnObject(const cv::Rect& ext, const char *n = NULL);
    //drwnObject(const vector<cv::Point>& polygon, const char *n = NULL);
    //! copy constructor
    drwnObject(const drwnObject& obj);
    ~drwnObject();

    //! save an object
    bool save(drwnXMLNode& xml) const;
    //! load an object
    bool load(drwnXMLNode& xml);

    //! return true if the point (x,y) is within the object
    bool hit(double x, double y) const;
    //! return true if the point \p pt is within the object
    bool hit(const cv::Point& pt) const { return hit(pt.x, pt.y); }

    //! return the area of the object
    double area() const { return fabs((double)(extent.width * extent.height)); }
    //! return the aspect ratio (width/height) of the object
    double aspect() const { return (double)extent.width / (double)extent.height; }

    //! return the number of pixels that overlap between the object and \p roi
    double overlap(const cv::Rect& roi) const;
    //! return the number of pixels that overlap between two objects
    double overlap(const drwnObject& obj) const { return overlap(obj.extent); }

    //! return the area of intersection divided by the area of union
    double areaOverlap(const cv::Rect& roi) const {
        const double ov = overlap(roi);
        return (ov / (area() + (roi.width * roi.height) - ov));
    }
    //! return the area of intersection divided by the area of union
    double areaOverlap(const drwnObject& obj) const { return areaOverlap(obj.extent); }

    //! return true if \p obj is within the object
    bool inside(const drwnObject& obj) const {
        return (fabs(overlap(obj) - area()) < DRWN_EPSILON);
    }

    //! scale the object bounding box
    void scale(double x_scale, double y_scale) {
        extent.x = (int)(x_scale * extent.x);
        extent.y = (int)(y_scale * extent.y);
        extent.width = (int)(x_scale * extent.width);
        extent.height = (int)(y_scale * extent.height);
    }
    //! scale the object bounding box
    void scale(double sc) { scale(sc, sc); }

    drwnObject& operator=(const drwnObject& obj);
    bool operator==(const drwnObject& obj) const;
    bool operator!=(const drwnObject& obj) const { return !(*this == obj); }
    bool operator<(const drwnObject& obj) const;
    bool operator>(const drwnObject& obj) const;
};

// drwnObjectList -----------------------------------------------------------
//! List of objects for the same image (see drwnObject)

class drwnObjectList : public drwnWriteable, public std::vector<drwnObject> {
 public:
    drwnObjectList();
    ~drwnObjectList();

    // i/o
    const char *type() const { return "drwnObjectList"; }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);

    //! sort all objects in the list by their score
    void sort();

    // filtering
    //! remove all objects whose name does not match \p name
    int removeNonMatching(const string& name);
    //! remove all objects whose name matches \p name
    int removeMatching(const string& name);
    //! perform non-maximal neighborhood suppression on the list of objects
    int nonMaximalSuppression(double threshold = 0.5, bool bByName = true);
    //! keep only the \p nKeep highest scoring objects in the list
    int keepHighestScoring(int nKeep);
    //! keep all objects in the list with score above \p threshold
    int keepAboveScore(double threshold);

    //! \todo split (by name) and merge functions
};

// drwnObjectSequence -------------------------------------------------------
//! Sequence of images, each with a list of objects (see drwnObjectList)

class drwnObjectSequence : public drwnWriteable, public std::map<string, drwnObjectList> {
 public:
    drwnObjectSequence();
    ~drwnObjectSequence();

    // i/o
    const char *type() const { return "drwnObjectSequence"; }
    bool save(drwnXMLNode& xml) const;
    bool load(drwnXMLNode& xml);
};

// utility functions
void printObjectStatistics(const drwnObjectSequence& objects);

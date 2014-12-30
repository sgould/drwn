/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDataExplorerNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Node for multi-dimensional data exploration.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

#include "wx/glcanvas.h"

using namespace std;

// drwnDataExplorerNode ------------------------------------------------------

class drwnDataExplorerNode : public drwnNode {
 protected:
    int _colour;                // colour of records for exploring
    int _subSamplingRate;       // subsampling rate
    bool _bIgnoreMissing;       // ignore missing labels
    int _maxPoints;             // maximum number of points to accumulate

    vector<vector<double> > _features; // data features
    vector<int> _labels;        // data labels (markers)

 public:
    drwnDataExplorerNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnDataExplorerNode(const drwnDataExplorerNode& node);
    virtual ~drwnDataExplorerNode();

    // i/o
    const char *type() const { return "drwnDataExplorerNode"; }
    drwnDataExplorerNode *clone() const { return new drwnDataExplorerNode(*this); }

    // gui
    void showWindow();
    void updateWindow();

    // processing
    void evaluateForwards();

    // learning
    void resetParameters();
};

// drwnDataExplorerCanvas ----------------------------------------------------

class drwnDataExplorerCanvas : public wxGLCanvas
{
 protected:
    vector<vector<double> > _data; // data to plot
    vector<int> _labels;           // marker colour
    vector<double> _mu;            // data centroid
    vector<double> _sigma;         // data variance

    double _pointSize;
    double _bShowCrossHairs;
    int _wndWidth, _wndHeight;
    wxPoint _lastMousePoint;

    // viewpoint parameters
    Eigen::Vector3d _cameraUp;
    Eigen::Vector3d _cameraPosition;
    Eigen::Vector3d _cameraTarget;
    double _cameraPanAngle;
    double _cameraTiltAngle;
    double _cameraDistance;

 public:
    drwnDataExplorerCanvas(wxWindow *parent, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize &size = wxDefaultSize,
        long style = 0, const wxString &name = wxT("drwnDataExplorerCanvas"));
    ~drwnDataExplorerCanvas();

    void onEraseBackground(wxEraseEvent &event);
    void onPaint(wxPaintEvent &event);
    void onSize(wxSizeEvent &event);
    void onKey(wxKeyEvent &event);
    void onMouse(wxMouseEvent &event);

    void updateCameraPosition();
    void setBackgroundColor(const wxColour &color);

    void clearData();
    void setData(const vector<vector<double> >& data);
    void setData(const vector<vector<double> >& data, const vector<int>& labels);

 protected:
    void cacheCentroid();
    void setView();
    void render();

    DECLARE_EVENT_TABLE()
};

// drwnDataExplorerWindow ----------------------------------------------------

class drwnDataExplorerWindow : public drwnGUIWindow
{
 protected:
    drwnDataExplorerCanvas *_canvas;

 public:
    drwnDataExplorerWindow(drwnNode *node);
    ~drwnDataExplorerWindow();

    drwnDataExplorerCanvas *canvas() const { return _canvas; }

    void onKey(wxKeyEvent &event);
    void onClose(wxCloseEvent& event);

 protected:
    DECLARE_EVENT_TABLE()
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnDataExplorerNode);

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGUIWindow.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Window control for Darwin GUI allowing nodes to display results in a
**  graphical format. Also implements some pre-defined GUI types (such as
**  scatter and bar plots, and interactive shells).
**
*****************************************************************************/

#pragma once

// wxWidgets
#include "wx/wx.h"
#include "wx/utils.h"
#include "wx/dialog.h"
#include "wx/textctrl.h"

using namespace std;

// forward declarations ------------------------------------------------------

class drwnNode;

// drwnGUIWindow -------------------------------------------------------------

class drwnGUIWindow : public wxDialog {
 protected:
    drwnNode *_owner;

 public:
    drwnGUIWindow(drwnNode *owner);
    virtual ~drwnGUIWindow();

    // callbacks
    void onClose(wxCloseEvent& event);

 protected:
    DECLARE_EVENT_TABLE()
};

// drwnGUIScatterPlot --------------------------------------------------------

class drwnGUIScatterPlot : public drwnGUIWindow {
 protected:
    vector<double> _x;
    vector<double> _y;
    vector<int> _labels;
    pair<double, double> _xRange;
    pair<double, double> _yRange;

 public:
    drwnGUIScatterPlot(drwnNode *owner);
    ~drwnGUIScatterPlot();

    void setData(const vector<double>& x, const vector<double>& y);
    void setData(const vector<double>& x, const vector<double>& y,
        const vector<int>& labels);
    void setXRange(double minX, double maxX);
    void setYRange(double minY, double maxY);

    // callbacks
    void onPaint(wxPaintEvent &event);

 protected:
    DECLARE_EVENT_TABLE()
};

// drwnGUIBarPlot ------------------------------------------------------------

class drwnGUIBarPlot : public drwnGUIWindow {
 protected:
    bool _bStacked;
    Eigen::MatrixXd _data;
    vector<string> _labels;

 public:
    drwnGUIBarPlot(drwnNode *owner);
    ~drwnGUIBarPlot();

    void setData(const Eigen::MatrixXd &data);
    void setLabels(const vector<string>& labels);
    void setStacked(bool bStacked = true);

    // callbacks
    void onPaint(wxPaintEvent &event);

 protected:
    DECLARE_EVENT_TABLE()
};

// drwnGUILinePlot -----------------------------------------------------------

class drwnGUILinePlot : public drwnGUIWindow {
 protected:
    list<vector<pair<double, double> > > _curves;
    list<int> _labels;
    pair<double, double> _xRange;
    pair<double, double> _yRange;

 public:
    drwnGUILinePlot(drwnNode *owner);
    ~drwnGUILinePlot();

    void clear();
    void addCurve(const vector<pair<double, double> >& c, int lbl = -1);
    void setXRange(double minX, double maxX);
    void setYRange(double minY, double maxY);

    // callbacks
    void onPaint(wxPaintEvent &event);

 protected:
    DECLARE_EVENT_TABLE()
};

// drwnGUIShellWindow --------------------------------------------------------

enum
{
    SHELL_WND_COMMAND_INPUT = wxID_HIGHEST + 10,
    SHELL_WND_CLEAR_OUTPUT = wxID_HIGHEST + 20,
    SHELL_WND_CLEAR_HISTORY = wxID_HIGHEST + 30
};

class drwnGUIShellWindow : public drwnGUIWindow {
 protected:
    vector<string> _history;
    int _currentHistory;
    wxTextCtrl *_shellOutput;
    wxTextCtrl *_shellInput;

 public:
    drwnGUIShellWindow(drwnNode *owner);
    ~drwnGUIShellWindow();

    // callbacks
    void onKeyPress(wxKeyEvent &event);
    void onEnter(wxCommandEvent &event);
    void onBtnClick(wxCommandEvent& event);

 protected:
    virtual bool executeCommand(const string& cmd) = 0;

    DECLARE_EVENT_TABLE()
};

// drwnGUIShowImageWindow ----------------------------------------------------

class drwnGUIShowImageWindow : public drwnGUIWindow {
 protected:
    list<wxImage> _views;

 public:
    drwnGUIShowImageWindow(drwnNode *owner);
    ~drwnGUIShowImageWindow();

    void clear();
    void addView(const wxImage& img);
    void addView(const unsigned char *data, int width, int height);

    // callbacks
    void onPaint(wxPaintEvent &event);

 protected:
    DECLARE_EVENT_TABLE()
};

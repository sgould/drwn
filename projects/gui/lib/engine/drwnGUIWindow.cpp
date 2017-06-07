/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGUIWindow.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "drwnBase.h"
#include "drwnEngine.h"

#include "wx/wxprec.h"
#include "wx/utils.h"
#include "wx/dcbuffer.h"
#include "wx/dialog.h"

#include "drwnGUIWindow.h"

using namespace std;

// Event Tables --------------------------------------------------------------

BEGIN_EVENT_TABLE(drwnGUIWindow, wxDialog)
    EVT_CLOSE(drwnGUIWindow::onClose)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnGUIScatterPlot, drwnGUIWindow)
    EVT_PAINT(drwnGUIScatterPlot::onPaint)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnGUIBarPlot, drwnGUIWindow)
    EVT_PAINT(drwnGUIBarPlot::onPaint)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnGUILinePlot, drwnGUIWindow)
    EVT_PAINT(drwnGUILinePlot::onPaint)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnGUIShellWindow, drwnGUIWindow)
    EVT_CHAR(drwnGUIShellWindow::onKeyPress)
    EVT_TEXT_ENTER(SHELL_WND_COMMAND_INPUT, drwnGUIShellWindow::onEnter)
    EVT_BUTTON(SHELL_WND_CLEAR_OUTPUT, drwnGUIShellWindow::onBtnClick)
    EVT_BUTTON(SHELL_WND_CLEAR_HISTORY, drwnGUIShellWindow::onBtnClick)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnGUIShowImageWindow, drwnGUIWindow)
    EVT_PAINT(drwnGUIShowImageWindow::onPaint)
END_EVENT_TABLE()

// drwnGUIWindow -------------------------------------------------------------

drwnGUIWindow::drwnGUIWindow(drwnNode *owner) :
    wxDialog(NULL, wxID_ANY, owner->getName().c_str(), wxDefaultPosition, wxDefaultSize,
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxFRAME_TOOL_WINDOW | wxFULL_REPAINT_ON_RESIZE),
    _owner(owner)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT); // for wxAutoBufferedPaintDC
}

drwnGUIWindow::~drwnGUIWindow()
{
    // do nothing
}

// callbacks
void drwnGUIWindow::onClose(wxCloseEvent& event)
{
    this->Hide();
    event.Skip();
}


// drwnGUIScatterPlot --------------------------------------------------------

drwnGUIScatterPlot::drwnGUIScatterPlot(drwnNode *owner) :
    drwnGUIWindow(owner)
{
    _xRange = make_pair(-1.0, 1.0);
    _yRange = make_pair(-1.0, 1.0);
}

drwnGUIScatterPlot::~drwnGUIScatterPlot()
{
    // do nothing
}

void drwnGUIScatterPlot::setData(const vector<double>& x, const vector<double>& y)
{
    DRWN_ASSERT(x.size() == y.size());
    setData(x, y, vector<int>());
}

void drwnGUIScatterPlot::setData(const vector<double>& x, const vector<double>& y,
    const vector<int>& labels)
{
    DRWN_ASSERT((x.size() == y.size()) && (labels.empty() || (labels.size() == x.size())));

    _x = x;
    _y = y;
    _labels = labels;

    if (_x.empty()) {
        _xRange = make_pair(-1.0, 1.0);
        _yRange = make_pair(-1.0, 1.0);
    } else {
        _xRange = drwn::range(_x);
        _yRange = drwn::range(_y);

        _xRange.first -= DRWN_EPSILON;
        _xRange.second += DRWN_EPSILON;
        _yRange.first -= DRWN_EPSILON;
        _yRange.second += DRWN_EPSILON;

        double d = _xRange.second - _xRange.first;
        _xRange.first -= 0.05 * d;
        _xRange.second += 0.05 * d;

        d = _yRange.second - _yRange.first;
        _yRange.first -= 0.05 * d;
        _yRange.second += 0.05 * d;
    }

    Refresh(false);
    Update();
}

void drwnGUIScatterPlot::setXRange(double minX, double maxX)
{
    DRWN_ASSERT(minX < maxX);
    _xRange.first = minX;
    _xRange.second = maxX;

    Refresh(false);
    Update();
}

void drwnGUIScatterPlot::setYRange(double minY, double maxY)
{
    DRWN_ASSERT(minY < maxY);
    _yRange.first = minY;
    _yRange.second = maxY;

    Refresh(false);
    Update();
}

// callbacks
void drwnGUIScatterPlot::onPaint(wxPaintEvent &event)
{
    int width, height;
    GetClientSize(&width, &height);

    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    // draw grid
#ifdef __LINUX__
    dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxSHORT_DASH));
#else
    dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxDOT));
#endif
    for (double v = 0.1; v < 1.0; v += 0.1) {
        dc.DrawLine(0, (int)(v * height), width, (int)(v * height));
        dc.DrawLine((int)(v * width), 0, (int)(v * width), height);
    }

    // draw points
    dc.SetPen(*wxRED_PEN);
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    for (int i = 0; i < (int)_x.size(); i++) {
        double u = (_x[i] - _xRange.first) / (_xRange.second - _xRange.first);
        double v = (_yRange.second - _y[i]) / (_yRange.second - _yRange.first);

        if (!_labels.empty()) {
            switch (_labels[i] % 6) {
            case 0: dc.SetPen(*wxRED_PEN); break;
            case 1: dc.SetPen(wxPen(*wxBLUE)); break;
            case 2: dc.SetPen(*wxGREEN_PEN); break;
            case 3: dc.SetPen(wxPen(wxColour(0xff, 0x00, 0xff))); break;
            case 4: dc.SetPen(*wxCYAN_PEN); break;
            case 5: dc.SetPen(wxPen(wxColour(0xff, 0xff, 0x00))); break;
            default: dc.SetPen(*wxBLACK_PEN); // for negative
            }
        }

        if (!_labels.empty()) {
            switch ((int)(_labels[i] / 6)) {
            case 0: // circle
                dc.DrawCircle((wxCoord)(width * u), (wxCoord)(height * v), 3);
                break;
            case 1: // box
                dc.DrawRectangle((wxCoord)(width * u) - 1, (wxCoord)(height * v) - 1, 3, 3);
                break;
            case 2: // cross
                dc.DrawLine((wxCoord)(width * u) - 1, (wxCoord)(height * v) - 1,
                    (wxCoord)(width * u) + 1, (wxCoord)(height * v) + 1);
                dc.DrawLine((wxCoord)(width * u) - 1, (wxCoord)(height * v) + 1,
                    (wxCoord)(width * u) + 1, (wxCoord)(height * v) - 1);
                break;
            default: // plus
                dc.DrawLine((wxCoord)(width * u) - 1, (wxCoord)(height * v),
                    (wxCoord)(width * u) + 1, (wxCoord)(height * v));
                dc.DrawLine((wxCoord)(width * u), (wxCoord)(height * v) - 1,
                    (wxCoord)(width * u), (wxCoord)(height * v) + 1);
            }
        } else {
            dc.DrawCircle((wxCoord)(width * u), (wxCoord)(height * v), 3);
        }
    }
}

// drwnGUIBarPlot ------------------------------------------------------------

drwnGUIBarPlot::drwnGUIBarPlot(drwnNode *owner) :
    drwnGUIWindow(owner), _bStacked(false)
{
    // do nothing
}

drwnGUIBarPlot::~drwnGUIBarPlot()
{
    // do nothing
}

void drwnGUIBarPlot::setData(const Eigen::MatrixXd &data)
{
    _data = data;
    if (_data.rows() != (int)_labels.size()) {
        _labels.clear();
    }

    Refresh(false);
    Update();
}

void drwnGUIBarPlot::setLabels(const vector<string>& labels)
{
    DRWN_ASSERT(labels.empty() || ((int)labels.size() == _data.rows()));

    _labels = labels;

    Refresh(false);
    Update();
}

void drwnGUIBarPlot::setStacked(bool bStacked)
{
    _bStacked = bStacked;

    Refresh(false);
    Update();
}

// callbacks
void drwnGUIBarPlot::onPaint(wxPaintEvent &event)
{
    int width, height;
    GetClientSize(&width, &height);

    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (_data.size() == 0)
        return;

    // plot properties
    int numGroups = _data.rows();
    int numBarsPerGroup = _data.cols();
    double minValue = _data.minCoeff();
    if (minValue < 0.0) {
        _bStacked = false;
    } else {
        minValue = 0.0;
    }
    double maxValue = _bStacked ? _data.rowwise().sum().maxCoeff() :
        std::max(0.0, _data.maxCoeff());
    if (maxValue == minValue) {
        maxValue += 1.0;
        minValue -= 1.0;
    }

    bool bHasText = !_labels.empty();

    int barWidth = (int)(0.9 * width / (numGroups + 1));
    if (barWidth < 10) return; // too small to draw

    if (!_bStacked) barWidth /= numBarsPerGroup;
    int barSpacing = barWidth;
    if (barWidth < 10) barWidth = 10;

    // build colour table (rainbow)
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    vector<wxColour> colourTable;
    for (int i = 0; i < numBarsPerGroup; i++) {
        int h = (int)(5 * (i + 2) / (numBarsPerGroup + 2));
        unsigned char color = (unsigned char)(255 * (5.0 * (i + 2) / (numBarsPerGroup + 2) - (double)h));
        switch (h) {
        case 0:
            red = 0x00; green = 0x00; blue = color; break;
        case 1:
            red = 0x00; green = color; blue = 0xff; break;
        case 2:
            red = 0x00; green = 0xff; blue = 0xff - color; break;
        case 3:
            red = color; green = 0xff; blue = 0x00; break;
        case 4:
            red = 0xff; green = 0xff - color; blue = 0x00; break;
        default:
            red = 0xff; green = 0x00; blue = 0x00; break;
        }
        colourTable.push_back(wxColour(red, green, blue));
    }

    // draw bars
    dc.SetPen(*wxBLACK_PEN);
    int u, v;
    int groupOffset = (int)(0.6 * width / (numGroups + 1));
    for (int i = 0; i < numGroups; i++) {
        double cumSum = 0.0;
        for (int j = 0; j < numBarsPerGroup; j++) {
            if (_bStacked) {
                u = groupOffset;
                v = (int)(height * (cumSum - minValue) / (maxValue - minValue));
            } else {
                u = groupOffset + j * barSpacing;
                v = (int)(height * (0.0 - minValue) / (maxValue - minValue));
            }

            int barHeight = (int)(height * _data(i, j) / (maxValue - minValue));
            dc.SetBrush(wxBrush(colourTable[j]));
            dc.DrawRectangle(u, height - v - barHeight, barWidth, barHeight);

            cumSum += _data(i, j);
        }

        groupOffset += width / (numGroups + 1);
    }

    // draw text
    // TODO
}


// drwnGUILinePlot -----------------------------------------------------------

drwnGUILinePlot::drwnGUILinePlot(drwnNode *owner) :
    drwnGUIWindow(owner)
{
    _xRange = make_pair(0.0, 1.0);
    _yRange = make_pair(0.0, 1.0);
}

drwnGUILinePlot::~drwnGUILinePlot()
{
    // do nothing
}

void drwnGUILinePlot::clear()
{
    _curves.clear();
    _labels.clear();
    _xRange = make_pair(0.0, 1.0);
    _yRange = make_pair(0.0, 1.0);
}

void drwnGUILinePlot::addCurve(const vector<pair<double, double> >& c, int lbl)
{
    _curves.push_back(c);
    _labels.push_back(lbl);

    Refresh(false);
    Update();
}

void drwnGUILinePlot::setXRange(double minX, double maxX)
{
    DRWN_ASSERT(minX < maxX);
    _xRange.first = minX;
    _xRange.second = maxX;

    Refresh(false);
    Update();
}

void drwnGUILinePlot::setYRange(double minY, double maxY)
{
    DRWN_ASSERT(minY < maxY);
    _yRange.first = minY;
    _yRange.second = maxY;

    Refresh(false);
    Update();
}

// callbacks
void drwnGUILinePlot::onPaint(wxPaintEvent &event)
{
    int width, height;
    GetClientSize(&width, &height);

    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    // draw grid
#ifdef __LINUX__
    dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxSHORT_DASH));
#else
    dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxDOT));
#endif
    for (double v = 0.1; v < 1.0; v += 0.1) {
        dc.DrawLine(0, (int)(v * height), width, (int)(v * height));
        dc.DrawLine((int)(v * width), 0, (int)(v * width), height);
    }

    // draw curves
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    list<vector<pair<double, double> > >::const_iterator it = _curves.begin();
    list<int>::const_iterator jt = _labels.begin();
    while (it != _curves.end()) {
        // skip curves with less than 2 points
        if (it->size() < 2) {
            it++; jt++; continue;
        }

        // set colour
        switch (*jt % 6) {
        case 0: dc.SetPen(wxPen(*wxRED, 2)); break;
        case 1: dc.SetPen(wxPen(*wxBLUE, 2)); break;
        case 2: dc.SetPen(wxPen(*wxGREEN, 2)); break;
        case 3: dc.SetPen(wxPen(wxColour(0xff, 0x00, 0xff), 2)); break;
        case 4: dc.SetPen(wxPen(*wxCYAN, 2)); break;
        case 5: dc.SetPen(wxPen(wxColour(0xff, 0xff, 0x00), 2)); break;
        default: dc.SetPen(*wxBLACK_PEN); // for negative
        }

        // draw curve
        double u0 = ((*it)[0].first - _xRange.first) / (_xRange.second - _xRange.first);
        double v0 = (_yRange.second - (*it)[0].second) / (_yRange.second - _yRange.first);
        for (unsigned i = 1; i < it->size(); i++) {
            double u1 = ((*it)[i].first - _xRange.first) / (_xRange.second - _xRange.first);
            double v1 = (_yRange.second - (*it)[i].second) / (_yRange.second - _yRange.first);
            if (((wxCoord)(u0 * width) != (wxCoord)(u1 * width)) ||
                ((wxCoord)(v0 * height) != (wxCoord)(v1 * height))) {
                dc.DrawLine((wxCoord)(u0 * width), (wxCoord)(v0 * height),
                    (wxCoord)(u1 * width), (wxCoord)(v1 * height));
            }
            u0 = u1; v0 = v1;
        }

        // iterate to next curve
        it++; jt++;
    }
}

// drwnGUIShellWindow --------------------------------------------------------

drwnGUIShellWindow::drwnGUIShellWindow(drwnNode *owner) : drwnGUIWindow(owner),
    _currentHistory(-1)
{
    wxBoxSizer *mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->SetMinSize(wxSize(320, 240));

    // create text control
    _shellOutput = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition,
        wxDefaultSize, wxEXPAND | wxTE_DONTWRAP | wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH);
    _shellInput = new wxTextCtrl(this, SHELL_WND_COMMAND_INPUT, "", wxDefaultPosition,
        wxDefaultSize, wxEXPAND | wxTE_DONTWRAP | wxTE_PROCESS_ENTER);

    // set fixed-width font
    wxTextAttr attr = _shellOutput->GetDefaultStyle();
    wxFont font(attr.GetFont());
    font.SetFaceName("Courier New");
    font.SetFamily(wxFONTFAMILY_MODERN);
    attr.SetFont(font);
    _shellOutput->SetDefaultStyle(attr);

    mainSizer->Add(_shellOutput, 1, wxEXPAND | wxALL, 10);
#if 1
    wxBoxSizer *ctrlSizer = new wxBoxSizer(wxHORIZONTAL);
    ctrlSizer->Add(_shellInput, 1, wxEXPAND | wxRIGHT, 10);
    ctrlSizer->Add(new wxButton(this, SHELL_WND_CLEAR_OUTPUT, "Clr &Output"), 0);
    ctrlSizer->Add(new wxButton(this, SHELL_WND_CLEAR_HISTORY, "Clr &History"), 0);

    mainSizer->Add(ctrlSizer, 0, wxEXPAND | wxBOTTOM | wxLEFT | wxRIGHT, 10);
#else
    mainSizer->Add(_shellInput, 0, wxEXPAND | wxBOTTOM | wxLEFT | wxRIGHT, 10);
#endif

    // implement the sizer
    SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);

    _shellInput->SetFocus();
}

drwnGUIShellWindow::~drwnGUIShellWindow()
{
    // do nothing
}

void drwnGUIShellWindow::onKeyPress(wxKeyEvent &event)
{
    // TODO: needs to process wxWANTS_CHARS to get this

    switch (event.m_keyCode) {
    case WXK_UP:
        if (!_history.empty()) {
            if (_currentHistory < 0) {
                _currentHistory = (int)_history.size() - 1;
            } else {
                _currentHistory -= 1;
                if (_currentHistory < 0) _currentHistory = 0;
            }
            _shellInput->SetValue(wxString(_history[_currentHistory]));
        }
        break;

    case WXK_DOWN:
        if (_currentHistory >= 0) {
            _currentHistory += 1;
            if (_currentHistory >= (int)_history.size()) {
                _currentHistory = -1;
                _shellInput->Clear();
            } else {
                _shellInput->SetValue(wxString(_history[_currentHistory]));
            }
        }
        break;

    default:
        event.Skip();
    }
}

void drwnGUIShellWindow::onEnter(wxCommandEvent &event)
{
    string cmd = string(_shellInput->GetValue().c_str());
    drwn::trim(cmd);
    if (!cmd.empty()) {
        if (_history.empty() || (_history.back() != cmd)) {
            _history.push_back(cmd);
        }

        // add command to output
        wxFont font = _shellOutput->GetDefaultStyle().GetFont();
        font.SetWeight(wxFONTWEIGHT_BOLD);
        _shellOutput->SetDefaultStyle(wxTextAttr(*wxBLUE, wxNullColour, font));
        _shellOutput->AppendText(cmd.c_str());
        _shellOutput->AppendText("\n");

        // execute the command
        font.SetWeight(wxFONTWEIGHT_NORMAL);
        _shellOutput->SetDefaultStyle(wxTextAttr(*wxBLACK, wxNullColour, font));
        executeCommand(cmd);
    }

    _shellInput->Clear();
    _currentHistory = -1;
}

void drwnGUIShellWindow::onBtnClick(wxCommandEvent &event)
{
    if (event.GetId() == SHELL_WND_CLEAR_OUTPUT) {
        _shellOutput->Clear();
    } else if (event.GetId() == SHELL_WND_CLEAR_HISTORY) {
        _history.clear();
        _currentHistory = -1;
    }

    event.Skip();
}

// drwnGUIShowImageWindow ----------------------------------------------------

drwnGUIShowImageWindow::drwnGUIShowImageWindow(drwnNode *owner) :
    drwnGUIWindow(owner)
{
    // do nothing
}

drwnGUIShowImageWindow::~drwnGUIShowImageWindow()
{
    // do nothing
}

void drwnGUIShowImageWindow::clear()
{
    _views.clear();

    Refresh(false);
    Update();
}

void drwnGUIShowImageWindow::addView(const wxImage& img)
{
    _views.push_back(img);

    Refresh(false);
    Update();
}

void drwnGUIShowImageWindow::addView(const unsigned char *data, int width, int height)
{
    wxImage img(width, height);
    unsigned char *p = img.GetData();
    // data format should be: RGBRGB...
    memcpy((void *)p, (void *)data, 3 * width * height);
    _views.push_back(img);

    Refresh(false);
    Update();
}

// callbacks
void drwnGUIShowImageWindow::onPaint(wxPaintEvent &event)
{
    int width, height;
    GetClientSize(&width, &height);

    wxPaintDC dc(this);
    if (_views.empty()) {
        dc.Clear();
        dc.SetTextForeground(wxColor(0, 0, 255));
        wxSize s = dc.GetTextExtent(wxT("no images"));
        dc.DrawText(wxT("no images"), (int)(width - s.x)/2, (int)(height - s.y)/2);
    } else {
        // TODO: draw multiple images
        dc.DrawBitmap(_views.back().Scale(width, height), 0, 0);
    }
}


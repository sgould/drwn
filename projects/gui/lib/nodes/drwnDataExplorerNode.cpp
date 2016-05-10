/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDataExplorerNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _UNICODE
#define UNICODE
#endif
#include <winsock2.h>
#endif
#include <cstdlib>

#ifdef __WXMAC__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnDataExplorerNode.h"

using namespace std;
using namespace Eigen;

// event tables --------------------------------------------------------------

BEGIN_EVENT_TABLE(drwnDataExplorerCanvas, wxGLCanvas)
  EVT_ERASE_BACKGROUND(drwnDataExplorerCanvas::onEraseBackground)
  EVT_SIZE(drwnDataExplorerCanvas::onSize)
  EVT_PAINT(drwnDataExplorerCanvas::onPaint)
  EVT_CHAR(drwnDataExplorerCanvas::onKey)
  EVT_MOUSE_EVENTS(drwnDataExplorerCanvas::onMouse)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnDataExplorerWindow, wxDialog)
  EVT_CHAR(drwnDataExplorerWindow::onKey)
  EVT_CLOSE(drwnDataExplorerWindow::onClose)
END_EVENT_TABLE()

// drwnDataExplorerNode ------------------------------------------------------

drwnDataExplorerNode::drwnDataExplorerNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _colour(-1), _subSamplingRate(1), _bIgnoreMissing(false),
    _maxPoints(1000000)
{
    _nVersion = 100;
    _desc = "Multi-dimensional data visualization";

    _inputPorts.push_back(new drwnInputPort(this, "dataIn",
            "N-by-K matrix of feature vectors"));
    _inputPorts.push_back(new drwnInputPort(this, "labelsIn",
            "N-by-K or N-by-1 matrix of class labels"));

    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
    declareProperty("maxPoints", new drwnRangeProperty(&_maxPoints, 0, DRWN_INT_MAX));
}

drwnDataExplorerNode::drwnDataExplorerNode(const drwnDataExplorerNode& node) :
    drwnNode(node), _colour(node._colour), _subSamplingRate(node._subSamplingRate),
    _bIgnoreMissing(node._bIgnoreMissing), _maxPoints(node._maxPoints)
{
    // declare propertys
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("subSample", new drwnRangeProperty(&_subSamplingRate, 1, DRWN_INT_MAX));
    declareProperty("ignoreMissing", new drwnBooleanProperty(&_bIgnoreMissing));
    declareProperty("maxPoints", new drwnRangeProperty(&_maxPoints, 0, DRWN_INT_MAX));
}

drwnDataExplorerNode::~drwnDataExplorerNode()
{
    // do nothing
}

// gui
void drwnDataExplorerNode::showWindow()
{
    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnDataExplorerWindow(this);
        _window->Show();
    }

    updateWindow();
}

void drwnDataExplorerNode::updateWindow()
{
    if ((_window == NULL) || (!_window->IsShown())) return;

    if (_features.empty() || (_features[0].size() < 2)) {
        ((drwnDataExplorerWindow *)_window)->canvas()->clearData();
        return;
    }

    ((drwnDataExplorerWindow *)_window)->canvas()->setData(_features, _labels);
}

// processing
void drwnDataExplorerNode::evaluateForwards()
{
    drwnDataTable *tblData = _inputPorts[0]->getTable();
    drwnDataTable *tblTarget = _inputPorts[1]->getTable();
    if (tblData == NULL) {
        DRWN_LOG_WARNING("node \"" << _name << "\" is missing data input");
        return;
    }

    _features.clear();
    _labels.clear();

    // iterate over input records
    vector<string> keys = tblData->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // skip missing targets or non-matching colour
        if ((tblTarget != NULL) && !tblTarget->hasKey(*it)) continue;
        if (!getOwner()->getDatabase()->matchColour(*it, _colour)) continue;

        // get data
        const drwnDataRecord *recData = tblData->lockRecord(*it);
        const drwnDataRecord *recTarget =
            (tblTarget == NULL) ? NULL : tblTarget->lockRecord(*it);

        // TODO: check data is the right format
        // target columns must be 1; rows must be 1 or data rows
        // (if exists) weight columns must be 1; rows must be 1 or data rows
        // data columns must match accumulated features

        // accumulate data (with sampling)
        for (int i = 0; i < recData->numObservations(); i++) {
            // ignore "unknown" class labels
            if (_bIgnoreMissing && (recTarget != NULL) && (recTarget->data()(i) < 0))
                continue;

            // random sample
            if ((_subSamplingRate > 1) && (rand() % _subSamplingRate != 0))
                continue;

            // add features
            _features.push_back(vector<double>(recData->numFeatures()));
            Eigen::Map<VectorXd>(&_features.back()[0], recData->numFeatures()) =
                recData->data().row(i);
            if (recTarget != NULL) {
                _labels.push_back((int)recTarget->data()(i));
            } else {
                _labels.push_back(-1);
            }

            // check haven't exceeded maximum number of points
            if (_features.size() >= (unsigned)_maxPoints)
                break;
        }

        // release records
        tblData->unlockRecord(*it);
        if (recTarget != NULL) tblTarget->unlockRecord(*it);

        // check haven't exceeded maximum number of points
        if (_features.size() >= (unsigned)_maxPoints)
            break;
    }

    DRWN_END_PROGRESS;
    updateWindow();
}

void drwnDataExplorerNode::resetParameters()
{
    _features.clear();
    _labels.clear();
    updateWindow();
}

// drwnDataExplorerCanvas ----------------------------------------------------

static int CANVAS_ARGS[] = { WX_GL_DOUBLEBUFFER, WX_GL_RGBA, 0 };

drwnDataExplorerCanvas::drwnDataExplorerCanvas(wxWindow *parent, wxWindowID id,
    const wxPoint& pos, const wxSize &size, long style, const wxString &name) :
    wxGLCanvas(parent, (wxGLCanvas *)NULL, id, pos, size,
        style | wxFULL_REPAINT_ON_RESIZE, name, CANVAS_ARGS),
    _pointSize(3.0), _bShowCrossHairs(true), _wndWidth(1), _wndHeight(1),
    _lastMousePoint(0, 0), _cameraUp(0.0, -1.0, 0.0),
    _cameraPosition(0.0, 0.0, 0.0), _cameraTarget(0.0, 0.0, 0.0),
    _cameraPanAngle(M_PI), _cameraTiltAngle(0.0), _cameraDistance(3.0)
{
    updateCameraPosition();
}

drwnDataExplorerCanvas::~drwnDataExplorerCanvas()
{
    // do nothing
}

void drwnDataExplorerCanvas::onEraseBackground(wxEraseEvent &event)
{
    // do nothing (and avoid flicker)
}

void drwnDataExplorerCanvas::onPaint(wxPaintEvent &event)
{
    wxPaintDC dc(this); // needed for MS Windows
    render();
}

void drwnDataExplorerCanvas::onSize(wxSizeEvent &event)
{
    wxGLCanvas::OnSize(event);
    GetClientSize(&_wndWidth, &_wndHeight);
    if (GetContext()) {
	this->setView();
    }
}

void drwnDataExplorerCanvas::onKey(wxKeyEvent &event)
{
    const double delta = 0.05;

    switch (event.m_keyCode) {
    case WXK_ESCAPE:
        this->GetParent()->Close();
	break;
    case 'a':
	_cameraTarget.y() += _cameraUp.y() * delta;
	break;
    case 'z':
	_cameraTarget.y() -= _cameraUp.y() * delta;
	break;
    case WXK_UP:
    case 'i':
	_cameraTarget.x() -= delta * _cameraDistance * sin(_cameraPanAngle);
	_cameraTarget.z() -= delta * _cameraDistance * cos(_cameraPanAngle);
        break;
    case WXK_DOWN:
    case 'k':
	_cameraTarget.x() += delta * _cameraDistance * sin(_cameraPanAngle);
	_cameraTarget.z() += delta * _cameraDistance * cos(_cameraPanAngle);
        break;
    case WXK_LEFT:
    case 'j':
	_cameraTarget.x() -= delta * _cameraUp.y() * _cameraDistance * cos(_cameraPanAngle);
	_cameraTarget.z() += delta * _cameraUp.y() * _cameraDistance * sin(_cameraPanAngle);
        break;
    case WXK_RIGHT:
    case 'l':
	_cameraTarget.x() += delta * _cameraUp.y() * _cameraDistance * cos(_cameraPanAngle);
	_cameraTarget.z() -= delta * _cameraUp.y() * _cameraDistance * sin(_cameraPanAngle);
        break;
    case 'x':
	_cameraDistance = 3.0;
        if (_sigma.size() >= 3) {
            _cameraDistance = 3.0 * sqrt(std::max(_sigma[0], std::max(_sigma[1], _sigma[2])));
        }

	_cameraPanAngle = M_PI;
	_cameraTiltAngle = 0.0;
#if 0
	if (bRightHandedCoordinates) {
	    _cameraUp << 0.0, -1.0, 0.0;
	} else {
	    _cameraUp << 0.0, 1.0, 0.0;
	}
#else
        _cameraUp << 0.0, -1.0, 0.0;
#endif
	_cameraTarget.setZero();
        if (_mu.size() >= 3) {
            _cameraTarget.x() = _mu[0];
            _cameraTarget.y() = _mu[1];
            _cameraTarget.z() = _mu[2];
        }
	break;
    case '+':
    case '=':
        _pointSize += 1.0;
        break;
    case '-':
    case '_':
        _pointSize -= 1.0;
        if (_pointSize < 1.0) _pointSize = 1.0;
        break;

    default:
        event.Skip();
    }

    updateCameraPosition();

    // refresh view
    this->Refresh(false);
    this->Update();
}

void drwnDataExplorerCanvas::onMouse(wxMouseEvent &event)
{
    int dx = event.m_x - _lastMousePoint.x;
    int dy = event.m_y - _lastMousePoint.y;

    _lastMousePoint = wxPoint(event.m_x, event.m_y);

    if (event.LeftIsDown() && event.Dragging())	{
	_cameraPanAngle -= 0.01 * _cameraUp.y() * dx;
#if 0
	if (_bRightHandedCoordinates)
	    _cameraTiltAngle += 0.01 * dy;
	else _cameraTiltAngle -= 0.01 * dy;
#else
        _cameraTiltAngle += 0.01 * dy;
#endif
	updateCameraPosition();
	this->Refresh(false);
	this->Update();

    } else if (event.RightIsDown() && event.Dragging()) {
	_cameraDistance *= (1.0 + 0.01 * dy);
	updateCameraPosition();
	this->Refresh(false);
	this->Update();
    }
}

void drwnDataExplorerCanvas::updateCameraPosition()
{
#if 0
    if (bRightHandedCoordinates) {
	_cameraUp << 0.0, -1.0, 0.0;
    } else {
	_cameraUp << 0.0, 1.0, 0.0;
    }
#endif

    _cameraPosition.x() = _cameraTarget.x() + _cameraDistance * cos(_cameraTiltAngle) * sin(_cameraPanAngle);
    _cameraPosition.y() = _cameraTarget.y() + _cameraUp.y() * _cameraDistance * sin(_cameraTiltAngle);
    _cameraPosition.z() = _cameraTarget.z() + _cameraDistance * cos(_cameraTiltAngle) * cos(_cameraPanAngle);
}

void drwnDataExplorerCanvas::setBackgroundColor(const wxColour &color)
{
    SetCurrent();
    glClearColor((float)color.Red() / 255.0f, (float)color.Green() / 255.0f,
        (float)color.Blue() / 255.0f, 0.0f);
}

void drwnDataExplorerCanvas::clearData()
{
    _data.clear();
    _labels.clear();
    cacheCentroid();

    this->Refresh(false);
    this->Update();
}

void drwnDataExplorerCanvas::setData(const vector<vector<double> >& data)
{
    _data = data;
    _labels.clear();
    cacheCentroid();

    this->Refresh(false);
    this->Update();
}

void drwnDataExplorerCanvas::setData(const vector<vector<double> >& data,
    const vector<int>& labels)
{
    DRWN_ASSERT(data.size() == labels.size());
    _data = data;
    _labels = labels;
    cacheCentroid();

    this->Refresh(false);
    this->Update();
}

void drwnDataExplorerCanvas::cacheCentroid()
{
    _mu.clear();
    _sigma.clear();

    if (_data.empty()) {
        return;
    }

    // accumulate
    _mu.resize(_data[0].size(), 0.0);
    _sigma.resize(_data[0].size(), 0.0);
    for (vector<vector<double> >::const_iterator it = _data.begin(); it != _data.end(); it++) {
        for (int i = 0; i < (int)_mu.size(); i++) {
            _mu[i] += (*it)[i];
            _sigma[i] += (*it)[i] * (*it)[i];
        }
    }

    // normalize
    for (int i = 0; i < (int)_mu.size(); i++) {
        _mu[i] /= (double)_data.size();
        _sigma[i] = std::max(_sigma[i] / (double)_data.size() - _mu[i] * _mu[i], DRWN_EPSILON);
    }

    DRWN_LOG_VERBOSE("drwnDataExplorer mean: " << toString(_mu));
    DRWN_LOG_VERBOSE("drwnDataExplorer variance: " << toString(_sigma));
}

void drwnDataExplorerCanvas::setView()
{
    SetCurrent();
    glViewport(0, 0, (GLint)_wndWidth, (GLint)_wndHeight);
    double aspectRatio = (double)_wndWidth / _wndHeight;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, aspectRatio, 0.1, 6000.0); // TODO: or glOrtho
    glMatrixMode(GL_MODELVIEW);

    glClearColor(0.0, 0.0, 0.0, 0.0); // background-color
}

void drwnDataExplorerCanvas::render()
{
    // initialization
    static bool _initialized = false;
    if (!_initialized) {
        setView();
        _initialized = true;
    }

    // clear buffer
    this->SetCurrent();

    // TODO move this to some initialization place
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // save current transformation
    glPushMatrix();
    gluLookAt(_cameraPosition.x(), _cameraPosition.y(), _cameraPosition.z(),
        _cameraTarget.x(), _cameraTarget.y(), _cameraTarget.z(),
        _cameraUp.x(), _cameraUp.y(), _cameraUp.z());

    // draw points
    if (!_data.empty() && (_data[0].size() >= 3)) {
	glColor3f(0.0, 1.0, 0.0);
	glPointSize(_pointSize);
	glBegin(GL_POINTS);
        if (!_labels.empty()) {
            for (unsigned i = 0; i < _data.size(); i++) {
                if (_labels[i] < 0) {
                    glColor3f(1.0, 1.0, 1.0);
                } else {
                    switch (_labels[i] % 6) {
                    case 0: glColor3f(1.0, 0.0, 0.0); break;
                    case 1: glColor3f(0.0, 1.0, 0.0); break;
                    case 2: glColor3f(0.0, 0.0, 1.0); break;
                    case 3: glColor3f(1.0, 1.0, 0.0); break;
                    case 4: glColor3f(1.0, 0.0, 1.0); break;
                    case 5: glColor3f(0.0, 1.0, 1.0); break;
                    default: glColor3f(1.0, 1.0, 1.0);
                    }
                    glVertex3f(_data[i][0], _data[i][1], _data[i][2]);
                }
            }
        } else {
            for (unsigned i = 0; i < _data.size(); i++) {
                glVertex3f(_data[i][0], _data[i][1], _data[i][2]);
            }
        }
	glEnd();
    }

    // draw look-at point
    if (_bShowCrossHairs) {
        glLineWidth(1.0);
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(_cameraTarget.x(), _cameraTarget.y(), _cameraTarget.z() - 0.05);
        glVertex3f(_cameraTarget.x(), _cameraTarget.y(), _cameraTarget.z() + 0.05);
        glColor3f(0.5, 0.5, 1.0);
        glVertex3f(_cameraTarget.x() - 0.05, _cameraTarget.y(), _cameraTarget.z());
        glVertex3f(_cameraTarget.x() + 0.05, _cameraTarget.y(), _cameraTarget.z());
        glColor3f(1.0, 1.0, 0.0);
        glVertex3f(_cameraTarget.x(), _cameraTarget.y() - 0.05, _cameraTarget.z());
        glVertex3f(_cameraTarget.x(), _cameraTarget.y() + 0.05, _cameraTarget.z());
        glEnd();
    }

    // restore previous transformation
    glPopMatrix();

    // draw and swap buffers
    glFlush();
    this->SwapBuffers();
}

// drwnDataExplorerWindow ----------------------------------------------------

drwnDataExplorerWindow::drwnDataExplorerWindow(drwnNode *node) :
    drwnGUIWindow(node), _canvas(NULL)
{
    wxBoxSizer *mainSizer = new wxBoxSizer(wxHORIZONTAL);
    _canvas = new drwnDataExplorerCanvas(this, wxID_ANY);
    mainSizer->Add(_canvas, 1, wxEXPAND | wxALL, 0);

    SetSizer(mainSizer);
    mainSizer->SetMinSize(wxSize(320, 240));
    mainSizer->SetSizeHints(this);
    mainSizer->Fit(this);

    _canvas->SetFocus();
}

drwnDataExplorerWindow::~drwnDataExplorerWindow()
{
    delete _canvas;
}

void drwnDataExplorerWindow::onKey(wxKeyEvent &event)
{
    if (_canvas != NULL)
        _canvas->onKey(event);
}

void drwnDataExplorerWindow::onClose(wxCloseEvent& event)
{
    // do nothing
    event.Skip();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Visualization", drwnDataExplorerNode);

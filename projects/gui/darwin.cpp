/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    darwin.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||(__VISUALC__)
#define _USE_MATH_DEFINES
#endif

// C++ Standard Headers
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <deque>

#if defined(__LINUX__)
#include <dlfcn.h>
#endif

// wxWidgets Headers
#include "wx/wx.h"
#include "wx/utils.h"
#include "wx/wxprec.h"
#include "wx/dcbuffer.h"
#include "wx/splitter.h"
#include "wx/aboutdlg.h"
#include "wx/dynlib.h"

#ifdef __WXMAC__
#include <ApplicationServices/ApplicationServices.h>
#endif

// Darwin Library Headers
#include "drwnBase.h"
#include "drwnEngine.h"

// Application Headers
#include "darwin.h"
#include "drwnIconFactory.h"
#include "drwnOptionsEditor.h"
#include "drwnTextEditor.h"

#include "resources/darwin.xpm"

using namespace std;

#define NOT_IMPLEMENTED_YET wxMessageBox("Functionality not implementet yet.", \
        "Error", wxOK | wxICON_EXCLAMATION, this);

// Global Variables and Tables -------------------------------------------------

MainWindow *gMainWindow = NULL;

// Event Tables ----------------------------------------------------------------

BEGIN_EVENT_TABLE(MainCanvas, wxWindow)
    EVT_ERASE_BACKGROUND(MainCanvas::on_erase_background)
    EVT_SIZE(MainCanvas::on_size)
    EVT_PAINT(MainCanvas::on_paint)
    EVT_CHAR(MainCanvas::on_key)
    EVT_MOUSE_EVENTS(MainCanvas::on_mouse)
    // node popup
    EVT_MENU(NODE_POPUP_SET_NAME, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_PROPERTIES, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_SHOWHIDE, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_EVALUATE, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_UPDATE, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_PROPBACK, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_RESETPARAMS, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_INITPARAMS, MainCanvas::on_popup_menu)
    EVT_MENU_RANGE(NODE_POPUP_INPORT_BASE, NODE_POPUP_INPORT_BASE + 100,
        MainCanvas::on_popup_menu)
    EVT_MENU_RANGE(NODE_POPUP_OUTPORT_BASE, NODE_POPUP_OUTPORT_BASE + 100,
        MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_DISCONNECT, MainCanvas::on_popup_menu)
    EVT_MENU(NODE_POPUP_DELETE, MainCanvas::on_popup_menu)
    // graph popup
    EVT_MENU(POPUP_SET_TITLE, MainCanvas::on_popup_menu)
    EVT_MENU_RANGE(POPUP_NODE_INSERT_BASE, POPUP_NODE_INSERT_BASE + 1000,
        MainCanvas::on_popup_menu)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(MainWindow, wxFrame)
    EVT_IDLE(MainWindow::on_idle)
    EVT_CLOSE(MainWindow::on_close)

    EVT_MENU(FILE_NEW, MainWindow::on_file_menu)
    EVT_MENU(FILE_OPEN, MainWindow::on_file_menu)
    EVT_MENU(FILE_SAVE, MainWindow::on_file_menu)
    EVT_MENU(FILE_SAVEAS, MainWindow::on_file_menu)
    EVT_MENU(FILE_CLOSE, MainWindow::on_file_menu)
    EVT_MENU(FILE_CLOSEALL, MainWindow::on_file_menu)
    EVT_MENU(FILE_EXPORT_HTML, MainWindow::on_file_menu)
    EVT_MENU(FILE_EXPORT_CODE, MainWindow::on_file_menu)
    EVT_MENU(FILE_EXPORT_SCRIPT, MainWindow::on_file_menu)
    EVT_MENU(FILE_EXIT, MainWindow::on_file_menu)

    EVT_MENU(EDIT_UNDO, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_REDO, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_CUT, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_COPY, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_PASTE, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_DELETE, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_FIND, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_SELECTALL, MainWindow::on_edit_menu)
    EVT_MENU(EDIT_DESELECTALL, MainWindow::on_edit_menu)

    EVT_MENU(NETWORK_RESET, MainWindow::on_network_menu)
    EVT_MENU(NETWORK_EVALUATE, MainWindow::on_network_menu)
    EVT_MENU(NETWORK_UPDATE, MainWindow::on_network_menu)
    EVT_MENU(NETWORK_BACKPROP, MainWindow::on_network_menu)
    EVT_MENU(NETWORK_INITPARAMS, MainWindow::on_network_menu)
    EVT_MENU(NETWORK_GRIDSNAP, MainWindow::on_network_menu)

    EVT_MENU(DATABASE_CONNECT, MainWindow::on_database_menu)
    EVT_MENU(DATABASE_VIEW_TABLES, MainWindow::on_database_menu)
    EVT_MENU(DATABASE_VIEW_INSTANCES, MainWindow::on_database_menu)
    EVT_MENU(DATABASE_IMPORT_COLOURS, MainWindow::on_database_menu)
    EVT_MENU(DATABASE_RANDOMIZE_COLOURS, MainWindow::on_database_menu)
    EVT_MENU(DATABASE_FLUSH_CACHE, MainWindow::on_database_menu)

    EVT_MENU(OPTIONS_DISPLAY_VERBOSE, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_DISPLAY_MESSAGE, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_DISPLAY_WARNING, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_BEEP, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_CLEAR_LOG, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_SAVE_LOG, MainWindow::on_options_menu)
    EVT_MENU(OPTIONS_TILE_WINDOWS, MainWindow::on_options_menu)

    EVT_MENU_RANGE(WINDOW_MENU_BASE, WINDOW_MENU_BASE + 99,
        MainWindow::on_window_menu)

    EVT_MENU(HELP_DRWN_CONTENTS, MainWindow::on_help_menu)
    EVT_MENU(HELP_RELEASE_NOTES, MainWindow::on_help_menu)
    EVT_MENU(HELP_ABOUT, MainWindow::on_help_menu)
END_EVENT_TABLE()

// GUI Callbacks ---------------------------------------------------------------

void messageCallback(const char *message)
{
    if (gMainWindow == NULL) {
        cout << "--- " << message << "\n";
    } else {
        if (drwnLogger::getLogLevel() >= DRWN_LL_DEBUG) {
            cout << "--- " << message << "\n";
        }
        gMainWindow->logMessage(message);
    }
}

void warningCallback(const char *message)
{
    if (drwnLogger::getLogLevel() >= DRWN_LL_DEBUG) {
        cerr << "-W- " << message << "\n";
    }
    gMainWindow->logMessage((string("WARNING: ") + string(message)).c_str(),
        wxTextAttr(*wxBLUE));
}

void errorCallback(const char *message)
{
    if (drwnLogger::getLogLevel() >= DRWN_LL_DEBUG) {
        cerr << "-E- " << message << "\n";
    }
    gMainWindow->logMessage((string("ERROR: ") + string(message)).c_str(),
        wxTextAttr(*wxRED));
}

void fatalCallback(const char *message)
{
    cerr << "-*- " << message << "\n";
    wxMessageBox(message, "Fatal Error", wxOK | wxICON_ERROR, NULL);
    exit(-1);
}

void progressCallback(const char *status, double progress)
{
    if (gMainWindow != NULL) {
        gMainWindow->updateProgress(status, progress);
    }
}

// MainCanvas Implementation ---------------------------------------------------

int MainCanvas::_creationCount = 0;

MainCanvas::MainCanvas(wxWindow *parent, wxWindowID id,const wxPoint& pos,
    const wxSize& size, long style, const wxString& name) :
    wxScrolledWindow(parent, id, pos, size, style, name), _graph(NULL),
    _activeNode(NULL), _activePort(NULL), _bSnapToGrid(true), _mouseMode(MM_NONE),
    _nodePopupMenu(NULL), _portSubMenuItem(NULL), _connectPopupMenu(NULL), 
    _newNodePopupMenu(NULL)
{
    _creationCount += 1;
    _graph = new drwnGraph(("New Network " + toString(_creationCount)).c_str());

    // initialize GUI elements
    this->SetBackgroundStyle(wxBG_STYLE_PAINT);
    this->SetBackgroundColour(*wxWHITE);
    //this->SetScrollbars(1, 1, 320, 240, 0, 0, true);
    this->SetVirtualSize(1024, 1024); // TODO: set this dynamically (size of graph + delta)
    this->SetScrollRate(1, 1);

    // create node popup menu
    _nodePopupMenu = new wxMenu();

    _nodePopupMenu->Append(NODE_POPUP_SET_NAME, "Set &Name...");
    _nodePopupMenu->Append(NODE_POPUP_PROPERTIES, "&Properties...");
    _nodePopupMenu->AppendSeparator();
    _nodePopupMenu->Append(NODE_POPUP_SHOWHIDE, "&Show...");
    _nodePopupMenu->AppendSeparator();
    _nodePopupMenu->Append(NODE_POPUP_EVALUATE, "&Evaluate Forwards");
    _nodePopupMenu->Append(NODE_POPUP_UPDATE, "&Update Forwards");
    _nodePopupMenu->Append(NODE_POPUP_PROPBACK, "&Propagate Backwards");
    _nodePopupMenu->Append(NODE_POPUP_RESETPARAMS, "&Reset Parameters");
    _nodePopupMenu->Append(NODE_POPUP_INITPARAMS, "&Estimate Parameters");
    _nodePopupMenu->AppendSeparator();
    _portSubMenuItem = _nodePopupMenu->AppendSubMenu(new wxMenu(), "Connect");
    _connectPopupMenu = new wxMenu();
    _nodePopupMenu->Append(NODE_POPUP_DISCONNECT, "Disconnect &All");
    _nodePopupMenu->AppendSeparator();
    _nodePopupMenu->Append(NODE_POPUP_DELETE, "&Delete");

    // create new node popup menu
    // TODO: put into own class
    _newNodePopupMenu = new wxMenu();

    _newNodePopupMenu->Append(POPUP_SET_TITLE, "Set &Title...");
    _newNodePopupMenu->AppendSeparator();

    int indx = 0;
    vector<string> groupNames = drwnNodeFactory::get().getGroups();
    for (vector<string>::const_iterator ig = groupNames.begin(); ig != groupNames.end(); ig++) {
        wxMenu *groupMenu = new wxMenu();
        // add nodes
        vector<string> nodeNames = drwnNodeFactory::get().getNodes(ig->c_str());
        for (vector<string>::const_iterator it = nodeNames.begin(); it != nodeNames.end(); it++) {
            string name = drwn::strReplaceSubstr(drwn::strSpacifyCamelCase(*it), string("drwn "), string());
            groupMenu->Append(POPUP_NODE_INSERT_BASE + indx, name);
            indx++;
        }

        // add to popup
        _newNodePopupMenu->AppendSubMenu(groupMenu, (string("Add ") + *ig).c_str());
    }
}

MainCanvas::~MainCanvas()
{
    // delete graph (database will be closed automtically)
    if (_graph != NULL) {
        delete _graph;
    }

    // delete gui elements
    if (_nodePopupMenu != NULL) {
        delete _nodePopupMenu;
    }
    if (_newNodePopupMenu != NULL) {
        delete _newNodePopupMenu;
    }
    if (_connectPopupMenu != NULL) {
        delete _connectPopupMenu;
    }
}

void MainCanvas::on_erase_background(wxEraseEvent &event)
{
    // do nothing (and avoid flicker)
}

void MainCanvas::on_paint(wxPaintEvent &WXUNUSED(event))
{
    const int SELWIDTH = 3;
    const int ARROWLEN = 7;
    const int ARROWWTH = 4;

    int width, height;
    GetVirtualSize(&width, &height);

    int clientX, clientY, clientWidth, clientHeight;
    GetViewStart(&clientX, &clientY);
    GetClientSize(&clientWidth, &clientHeight);

    //wxPaintDC dc(this);
    wxAutoBufferedPaintDC dc(this);
    DoPrepareDC(dc); // for scroll window
    dc.Clear();
    dc.SetFont(wxSystemSettings::GetFont(wxSYS_DEFAULT_GUI_FONT));

    // draw grid
    if (_bSnapToGrid) {
#ifdef __LINUX__
        dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxSHORT_DASH));
#else
        dc.SetPen(wxPen(*wxLIGHT_GREY, 1, wxDOT));
#endif
        for (int y = 16; y < height - 1; y += 32) {
            dc.DrawLine(0, y, width, y);
        }
        for (int x = 16; x < width - 1; x += 32) {
            dc.DrawLine(x, 0, x, height);
        }
    }

    DRWN_ASSERT(_graph != NULL);

    // draw title (TODO: add titlebar?)
    dc.SetTextForeground(wxColor(0, 0, 255));
    wxSize s = dc.GetTextExtent(_graph->getTitle().c_str());
    dc.DrawText(_graph->getTitle().c_str(), (int)(clientWidth - s.x)/2, 0);

    // draw arrows
    for (int i = 0; i < _graph->numNodes(); i++) {
        const drwnNode *node = _graph->getNode(i);
        const wxBitmap *nodeIcon = gIconFactory.getIcon(node->type());
        int tx = node->getLocationX() + nodeIcon->GetWidth()/2;
        int ty = node->getLocationY() + nodeIcon->GetHeight()/2;

        for (int j = 0; j < node->numInputPorts(); j++) {
            const drwnInputPort *port = node->getInputPort(j);
            if (port->getSource() == NULL) continue;
            const drwnNode *dstNode = port->getSource()->getOwner();
            if (dstNode == NULL) continue;

            int sx = dstNode->getLocationX() + nodeIcon->GetWidth()/2;
            int sy = dstNode->getLocationY() + nodeIcon->GetHeight()/2;

            double dx = tx - sx;
            double dy = ty - sy;
            double len = sqrt(dx * dx + dy * dy);
            dx /= len;
            dy /= len;

            int ddx = (int)(nodeIcon->GetWidth() * dx * M_SQRT1_2);
            int ddy = (int)(nodeIcon->GetHeight() * dy * M_SQRT1_2);
            dc.SetPen(wxPen(*wxBLACK, 2));
            dc.DrawLine((int)(sx + ddx), (int)(sy + ddy), (int)(tx - ddx), (int)(ty - ddy));

            wxPoint arrow[3];
            arrow[0] = wxPoint((int)(tx - ddx), (int)(ty - ddy));
            arrow[1] = wxPoint((int)(tx - ddx - ARROWLEN * dx + ARROWWTH * dy),
                (int)(ty - ddy - ARROWLEN * dy - ARROWWTH * dx));
            arrow[2] = wxPoint((int)(tx - ddx - ARROWLEN * dx - ARROWWTH * dy),
                (int)(ty - ddy - ARROWLEN * dy + ARROWWTH * dx));

            dc.SetPen(wxPen(*wxBLACK, 1));
            dc.SetBrush(*wxBLACK_BRUSH);
            dc.DrawPolygon(3, arrow);
        }
    }

    // draw nodes
    wxBrush selBrush(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT), wxSOLID);
    for (int i = 0; i < _graph->numNodes(); i++) {
        drwnNode *node = _graph->getNode(i);
        int nx = node->getLocationX();
        int ny = node->getLocationY();
        bool bSelected = (_selectedNodes.find(node) != _selectedNodes.end()) &&
            (_mouseMode != MM_DRAGGING);

        // draw bitmap
        const wxBitmap *nodeIcon = gIconFactory.getIcon(node->type());
        if (bSelected) {
            dc.SetPen(*wxTRANSPARENT_PEN);
            dc.SetBrush(selBrush);
            dc.DrawRectangle(nx - SELWIDTH, ny - SELWIDTH,
                nodeIcon->GetWidth() + 2 * SELWIDTH, nodeIcon->GetHeight() + 2 * SELWIDTH);
        }
        dc.DrawBitmap(*nodeIcon, nx, ny, true);

        // draw text
        vector<string> nameTokens;
        drwn::parseString(node->getName(), nameTokens);
        DRWN_ASSERT_MSG(!nameTokens.empty(), "\"" << node->getName() << "\"");

        vector<string> lines;
        lines.push_back(nameTokens[0]);
        for (int i = 1; i < (int)nameTokens.size(); i++) {
            wxSize s = dc.GetTextExtent(lines.back() + string(" ") + nameTokens[i]);
            if (s.x > 3 * nodeIcon->GetWidth()) {
                if (!bSelected && (lines.size() == 2)) {
                    lines.push_back(string("..."));
                    break;
                }
                lines.push_back(nameTokens[i]);
            } else {
                lines.back() += string(" ") + nameTokens[i];
            }
        }

        if (!bSelected) {
            for (int i = 0; i < (int)lines.size(); i++) {
                s = dc.GetTextExtent(lines[i]);
                if (s.x > 3 * nodeIcon->GetWidth()) {
                    while (s.x > 3 * nodeIcon->GetWidth()) {
                        lines[i].resize(lines[i].length() - 1);
                        s = dc.GetTextExtent(lines[i] + string("..."));
                    }
                    lines[i] = lines[i] + string("...");
                }
            }
        }

        if (bSelected) {
            int maxExtentX = 0;
            for (int i = 0; i < (int)lines.size(); i++) {
                s = dc.GetTextExtent(lines[i]);
                maxExtentX = std::max(maxExtentX, s.x);
            }

            dc.SetPen(*wxTRANSPARENT_PEN);
            dc.SetBrush(selBrush);
            dc.DrawRectangle(nx + (nodeIcon->GetWidth() - maxExtentX)/2 - SELWIDTH,
                ny + nodeIcon->GetHeight(),
                maxExtentX + 2 * SELWIDTH, lines.size() * s.y + SELWIDTH);
        }

        // boundary around text
        dc.SetTextForeground(bSelected ? wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT) :
            dc.GetTextBackground());
        for (int i = 0; i < (int)lines.size(); i++) {
            s = dc.GetTextExtent(lines[i]);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 - 1,
                ny + nodeIcon->GetHeight() + i * s.y + 2 - 1);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 + 1,
                ny + nodeIcon->GetHeight() + i * s.y + 2 + 1);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 - 1,
                ny + nodeIcon->GetHeight() + i * s.y + 2 + 1);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 + 1,
                ny + nodeIcon->GetHeight() + i * s.y + 2 - 1);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 - 2,
                ny + nodeIcon->GetHeight() + i * s.y + 2 - 2);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 + 2,
                ny + nodeIcon->GetHeight() + i * s.y + 2 + 2);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 - 2,
                ny + nodeIcon->GetHeight() + i * s.y + 2 + 2);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2 + 2,
                ny + nodeIcon->GetHeight() + i * s.y + 2 - 2);
        }

        //visible text
        dc.SetTextForeground(bSelected ?
            wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT) : wxColor(0, 0, 0));
        for (int i = 0; i < (int)lines.size(); i++) {
            s = dc.GetTextExtent(lines[i]);
            dc.DrawText(lines[i].c_str(), nx + (nodeIcon->GetWidth() - s.x)/2,
                ny + nodeIcon->GetHeight() + i * s.y + 2);
        }
    }

    // draw connection
    // TODO: move to on_mouse event like selection rubber-band
    if ((_mouseMode == MM_CONNECTING_INPUT) || (_mouseMode == MM_CONNECTING_OUTPUT)) {
        drwnNode *node = _activePort->getOwner();
        dc.SetPen(wxPen(*wxRED, 2));
        dc.DrawLine(node->getLocationX() + 32/2, node->getLocationY() + 32/2,
            _lastMousePoint.x, _lastMousePoint.y);
    }
}

void MainCanvas::on_size(wxSizeEvent &event)
{
    int width, height;

    GetClientSize(&width, &height);

    this->Refresh(false);
    this->Update();
}

void MainCanvas::on_key(wxKeyEvent &event)
{
    switch (event.m_keyCode) {
    case WXK_TAB:
    {
        if (_graph->numNodes() > 0) {
            int indx = 0;
            if (!_selectedNodes.empty()) {
                // find currently selected node and move to next one
                while (_graph->getNode(indx) != *_selectedNodes.begin()) {
                    indx++;
                }
                if (event.m_shiftDown) {
                    indx = (indx + _graph->numNodes() - 1) % _graph->numNodes();
                } else {
                    indx = (indx + 1) % _graph->numNodes();
                }
            }
            _selectedNodes.clear();
            _selectedNodes.insert(_graph->getNode(indx));
        }
        break;
    }
    default:
    	event.Skip();
    }

    // refresh view
    updateStatusBar();
    this->Refresh(false);
    this->Update();
}

void MainCanvas::on_mouse(wxMouseEvent &event)
{
    //_mousePoint = wxPoint(event.m_x, event.m_y);
    CalcUnscrolledPosition(event.m_x, event.m_y, &_mousePoint.x, &_mousePoint.y);

    if (event.LeftDown()) {
        //_buttonDownPoint = wxPoint(event.m_x, event.m_y);
        CalcUnscrolledPosition(event.m_x, event.m_y, &_buttonDownPoint.x, &_buttonDownPoint.y);

        // cancel connecting
        if ((_mouseMode == MM_CONNECTING_INPUT) || (_mouseMode == MM_CONNECTING_OUTPUT)) {
            if ((_activeNode == NULL) || (_activeNode == _activePort->getOwner())) {
                if (_mouseMode == MM_CONNECTING_INPUT) {
                    ((drwnInputPort *)_activePort)->disconnect();
                } else {
                    ((drwnOutputPort *)_activePort)->disconnect();
                }
                _mouseMode = MM_NONE;
                _activePort = NULL;
            } else if (_activeNode != NULL) {
                updatePortSubMenu(_activeNode);
                PopupMenu(_connectPopupMenu);
            }
        }
    } else if (event.LeftUp() && (_mouseMode == MM_NONE)) {
        // clicked on node or in empty space?
        if (_activeNode != NULL) {
            // node not already selected?
            if (_selectedNodes.find(_activeNode) == _selectedNodes.end()) {
                if (!event.ShiftDown() && !event.ControlDown()) {
                    _selectedNodes.clear();
                }
                _selectedNodes.insert(_activeNode);
            } else {
                if (!event.ShiftDown() && !event.ControlDown()) {
                    if (_selectedNodes.size() > 1) {
                        _selectedNodes.clear();
                        _selectedNodes.insert(_activeNode);
                    } else {
                        _selectedNodes.clear();
                    }
                } else {
                    _selectedNodes.erase(_activeNode);
                }
            }
        } else if (!event.ShiftDown()) {
            _selectedNodes.clear();
        }

        // refresh display
        this->Refresh(false);
        this->Update();

    } else if (event.LeftUp()) {
        if ((_mouseMode == MM_SELECTING) || (_mouseMode == MM_SELECTING_NOERASE)) {
            // finished selecting
            if (!event.ShiftDown()) {
                _selectedNodes.clear();
            }

            selectNodesInRegion(_buttonDownPoint.x, _buttonDownPoint.y,
                _lastMousePoint.x - _buttonDownPoint.x, _lastMousePoint.y - _buttonDownPoint.y);
            _mouseMode = MM_NONE;

        } else if (_mouseMode == MM_DRAGGING) {
            // finished moving
            if (_bSnapToGrid) {
                for (set<drwnNode *>::iterator it = _selectedNodes.begin();
                     it != _selectedNodes.end(); it++) {
                    int nx = 32 * ((int)((*it)->getLocationX() + 16) / 32);
                    int ny = 32 * ((int)((*it)->getLocationY() + 16) / 32);
                    (*it)->setLocation(nx, ny);
                }
            }
            _mouseMode = MM_NONE;
        }

        // refresh display
        this->Refresh(false);
        this->Update();

    } else if ((event.Entering() || event.Leaving()) && (_mouseMode == MM_SELECTING)) {
        // refresh display
        this->Refresh(false);
        this->Update();
        _mouseMode = MM_SELECTING_NOERASE;

    } else if (event.Dragging()) {
        if (_activeNode == NULL) {
            // selecting
            wxClientDC dc(this); // since not in on_paint event
            DoPrepareDC(dc);
            dc.SetPen(*wxBLACK_PEN);
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.SetLogicalFunction(wxINVERT);
            if (_mouseMode == MM_SELECTING) {
                dc.DrawRectangle(_buttonDownPoint.x, _buttonDownPoint.y,
                    _lastMousePoint.x - _buttonDownPoint.x, _lastMousePoint.y - _buttonDownPoint.y);
            }
            dc.DrawRectangle(_buttonDownPoint.x, _buttonDownPoint.y,
                _mousePoint.x - _buttonDownPoint.x, _mousePoint.y - _buttonDownPoint.y);
            _mouseMode = MM_SELECTING;
        } else {
            // node not already selected?
            if (_mouseMode != MM_DRAGGING) {
                if (_selectedNodes.find(_activeNode) == _selectedNodes.end()) {
                    if (!event.ShiftDown()) {
                        _selectedNodes.clear();
                    }
                    _selectedNodes.insert(_activeNode);
                }
            }

            // move
            for (set<drwnNode *>::iterator it = _selectedNodes.begin();
                 it != _selectedNodes.end(); it++) {
                (*it)->setLocation((*it)->getLocationX() + _mousePoint.x - _lastMousePoint.x,
                    (*it)->getLocationY() + _mousePoint.y - _lastMousePoint.y);
            }
            _mouseMode = MM_DRAGGING;
            this->Refresh(false);
            this->Update();
        }

    } else if (event.RightDown()) {
        if (_activeNode == NULL) {
            // insert a new node
            PopupMenu(_newNodePopupMenu);
        } else {
            // execute function on existing node
            updatePortSubMenu(_activeNode);
            if ((_mouseMode == MM_CONNECTING_INPUT) ||
                (_mouseMode == MM_CONNECTING_OUTPUT)) {
                PopupMenu(_connectPopupMenu);
            } else {
                PopupMenu(_nodePopupMenu);
            }
        }

    } else if (event.LeftDClick()) {
        if (_activeNode != NULL) {
            drwnOptionsEditor dlg(this, _activeNode);
            if (dlg.ShowModal() == wxID_OK) {
                this->Refresh(false);
                this->Update();
            }
        }

    } else if (event.Moving()) {
        int indx = findNodeAtLocation(_mousePoint.x, _mousePoint.y);
        _activeNode = (indx < 0) ? NULL : _graph->getNode(indx);
    }

    // scroll
    int delta = event.GetWheelRotation();
    if (delta != 0) {
        int clientX, clientY;
        this->GetViewStart(&clientX, &clientY);
        this->Scroll(clientX, std::max(0, clientY - delta));
    }

    // re-draw
    if ((_mouseMode == MM_CONNECTING_INPUT) ||
        (_mouseMode == MM_CONNECTING_OUTPUT)) {
        this->Refresh(false);
        this->Update();
    }

    //_lastMousePoint = wxPoint(event.m_x, event.m_y);
    _lastMousePoint = _mousePoint;
    updateStatusBar();
}

void MainCanvas::on_popup_menu(wxCommandEvent &event)
{
    if (event.GetId() == NODE_POPUP_SET_NAME) {
        DRWN_ASSERT(_activeNode != NULL);
        drwnNode *a = _activeNode; // save active node (which may change after dialog clicked)
        wxTextEntryDialog dlg(this, "Set the node's name:", "Set Name", _activeNode->getName());
        if (dlg.ShowModal() == wxID_OK) {
            a->setName(dlg.GetValue().ToStdString());
        }

    } else if (event.GetId() == NODE_POPUP_PROPERTIES) {
        drwnOptionsEditor dlg(this, _activeNode);
        if (dlg.ShowModal() == wxID_OK) {
            this->Refresh(false);
            this->Update();
        }

    } else if (event.GetId() == NODE_POPUP_SHOWHIDE) {
        if (_activeNode->isShowingWindow()) {
            _activeNode->hideWindow();
        } else {
            _activeNode->showWindow();
        }

    } else if (event.GetId() == NODE_POPUP_EVALUATE) {
        drwnLogger::setRunning(true);
        _activeNode->initializeForwards();
        _activeNode->evaluateForwards();
        _activeNode->finalizeForwards();
        drwnLogger::setRunning(false);

    } else if (event.GetId() == NODE_POPUP_UPDATE) {
        drwnLogger::setRunning(true);
        _activeNode->initializeForwards(false);
        _activeNode->updateForwards();
        _activeNode->finalizeForwards();
        drwnLogger::setRunning(false);

    } else if (event.GetId() == NODE_POPUP_RESETPARAMS) {
        _activeNode->resetParameters();

    } else if (event.GetId() == NODE_POPUP_INITPARAMS) {
        drwnLogger::setRunning(true);
        _activeNode->initializeParameters();
        drwnLogger::setRunning(false);

    } else if ((event.GetId() >= NODE_POPUP_INPORT_BASE) &&
        (event.GetId() < NODE_POPUP_INPORT_BASE + 100)) {
        int portId = event.GetId() - NODE_POPUP_INPORT_BASE;
        if (_mouseMode == MM_CONNECTING_OUTPUT) {
            DRWN_ASSERT(_activePort != NULL);
            _activeNode->getInputPort(portId)->connect((drwnOutputPort *)_activePort);
            _activeNode = NULL;
            _mouseMode = MM_NONE;
        } else {
            _activePort = _activeNode->getInputPort(portId);
            _mouseMode = MM_CONNECTING_INPUT;
        }

    } else if ((event.GetId() >= NODE_POPUP_OUTPORT_BASE) &&
        (event.GetId() < NODE_POPUP_OUTPORT_BASE + 100)) {
        int portId = event.GetId() - NODE_POPUP_OUTPORT_BASE;
        if (_mouseMode == MM_CONNECTING_INPUT) {
            DRWN_ASSERT(_activePort != NULL);
            _activeNode->getOutputPort(portId)->connect((drwnInputPort *)_activePort);
            _activeNode = NULL;
            _mouseMode = MM_NONE;
        } else {
            _activePort = _activeNode->getOutputPort(portId);
            _mouseMode = MM_CONNECTING_OUTPUT;
        }

    } else if (event.GetId() == NODE_POPUP_DISCONNECT) {
        // disconnect input ports
        for (int i = 0; i < _activeNode->numInputPorts(); i++) {
            _activeNode->getInputPort(i)->disconnect();
        }

        // disconnect output ports
        for (int i = 0; i < _activeNode->numOutputPorts(); i++) {
            _activeNode->getOutputPort(i)->disconnect();
        }

    } else if (event.GetId() == NODE_POPUP_DELETE) {
        DRWN_ASSERT(_activeNode != NULL);
        drwnNode *a = _activeNode; // save active node (which may change after dialog clicked)
        wxMessageDialog dlg(this, string("Delete node ") + a->getName() + string("?"),
            "Confirm", wxYES_NO | wxICON_QUESTION);
        if (dlg.ShowModal() == wxID_YES) {
            _graph->delNode(a);
            if (_activeNode == a)
                _activeNode = NULL;
            if (_selectedNodes.find(a) != _selectedNodes.end()) {
                _selectedNodes.erase(a);
            }
        }

    } else if (event.GetId() == POPUP_SET_TITLE) {
        wxTextEntryDialog dlg(this, "Set title for the data flow network:",
            "Set Title", _graph->getTitle());
        if (dlg.ShowModal() == wxID_OK) {
            _graph->setTitle(dlg.GetValue().ToStdString());
        }

    } else if (event.GetId() >= POPUP_NODE_INSERT_BASE) {
        // TODO: retrieve node group and name more efficiently
        DRWN_LOG_DEBUG("node insert popup selected (id: " <<
            (event.GetId() - POPUP_NODE_INSERT_BASE) << ")");

        drwnNode *node = NULL;
        int indx = event.GetId() - POPUP_NODE_INSERT_BASE;
        vector<string> groupNames = drwnNodeFactory::get().getGroups();
        for (vector<string>::const_iterator ig = groupNames.begin(); ig != groupNames.end(); ig++) {
            vector<string> nodeNames = drwnNodeFactory::get().getNodes(ig->c_str());
            for (vector<string>::const_iterator it = nodeNames.begin(); it != nodeNames.end(); it++) {
                if (--indx < 0) {
                    DRWN_LOG_VERBOSE("Creating new node of type \""
                        << it->c_str() << "\" from group \"" << ig->c_str() << "\"");
                    node = drwnNodeFactory::get().create(it->c_str());
                    break;
                }
            }
            if (node != NULL) break;
        }
        DRWN_ASSERT(node != NULL);

        if (_bSnapToGrid) {
            int nx = 32 * ((int)(_mousePoint.x + 16) / 32);
            int ny = 32 * ((int)(_mousePoint.y + 16) / 32);
            node->setLocation(nx, ny);
        } else {
            node->setLocation(_mousePoint.x, _mousePoint.y);
        }
        _graph->addNode(node); // unique node will be set by graph
    }

    this->Refresh(false);
    this->Update();
}

// select nodes within rectangle
void MainCanvas::selectNodesInRegion(int x, int y, int w, int h)
{
    // correct for negative widths and heights
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }

    // TODO: get correct icon size
    for (int i = 0; i < _graph->numNodes(); i++) {
        drwnNode *node = _graph->getNode(i);
        if ((x < node->getLocationX()) && (x + w > node->getLocationX() + 32) &&
            (y < node->getLocationY()) && (y + h > node->getLocationY() + 32)) {
            _selectedNodes.insert(node);
        }
    }
    this->Refresh(false);
    this->Update();
}

void MainCanvas::selectAllNodes()
{
    if ((int)_selectedNodes.size() == _graph->numNodes())
        return;

    for (int i = 0; i < _graph->numNodes(); i++) {
        _selectedNodes.insert(_graph->getNode(i));
    }
    this->Refresh(false);
    this->Update();
}

void MainCanvas::deselectAllNodes()
{
    if (_selectedNodes.empty())
        return;

    _selectedNodes.clear();
    this->Refresh(false);
    this->Update();
}

void MainCanvas::undoableAction()
{
    // TODO
}

void MainCanvas::updateImageBuffer()
{
    // TODO
}

void MainCanvas::updateStatusBar()
{
    if (_activeNode != NULL) {
        gMainWindow->_statusBar->updateMessage(_activeNode->getName().c_str());
        gMainWindow->_statusBar->SetStatusText(_activeNode->getDescription(), 3);
    } else {
        gMainWindow->_statusBar->updateMessage("");
        gMainWindow->_statusBar->SetStatusText("", 3);
    }
}

int MainCanvas::findNodeAtLocation(int x, int y) const
{
    DRWN_ASSERT(_graph != NULL);
    for (int i = 0; i < _graph->numNodes(); i++) {
        const drwnNode *node = _graph->getNode(i);
        int nx = node->getLocationX();
        int ny = node->getLocationY();
        if ((x > nx) && (x < nx + 32) && (y > ny) && (y < ny + 32)) {
            return i;
        }
    }

    return -1;
}

void MainCanvas::updatePortSubMenu(const drwnNode *node)
{
    DRWN_ASSERT((node != NULL) && (_portSubMenuItem != NULL));
    DRWN_ASSERT(_portSubMenuItem->GetSubMenu() != NULL);

    // show/hide
    if (node->isShowingWindow()) {
        _nodePopupMenu->SetLabel(NODE_POPUP_SHOWHIDE, "&Hide");
    } else {
        _nodePopupMenu->SetLabel(NODE_POPUP_SHOWHIDE, "&Show...");
    }

    // port submenu
#if 0
    delete _portSubMenuItem->GetSubMenu();

    wxMenu *portMenu;
    _portSubMenuItem->SetSubMenu(portMenu = new wxMenu());
#else
    // TODO: hack?
    wxMenu *portMenu = _portSubMenuItem->GetSubMenu();
    while (portMenu->GetMenuItemCount() > 0) {
        portMenu->Destroy(portMenu->FindItemByPosition(0));
    }
#endif
    delete _connectPopupMenu;
    _connectPopupMenu = new wxMenu();

    for (int i = 0; i < node->numInputPorts(); i++) {
        portMenu->Append(NODE_POPUP_INPORT_BASE + i,
            node->getInputPort(i)->getName());
        _connectPopupMenu->Append(NODE_POPUP_INPORT_BASE + i,
            node->getInputPort(i)->getName());
        if (_mouseMode == MM_CONNECTING_INPUT) {
            portMenu->Enable(NODE_POPUP_INPORT_BASE + i, false);
            _connectPopupMenu->Enable(NODE_POPUP_INPORT_BASE + i, false);
        }
    }

    if ((node->numInputPorts() > 0) && (node->numOutputPorts() > 0)) {
        portMenu->AppendSeparator();
        _connectPopupMenu->AppendSeparator();
    }

    for (int i = 0; i < node->numOutputPorts(); i++) {
        portMenu->Append(NODE_POPUP_OUTPORT_BASE + i,
            node->getOutputPort(i)->getName());
        _connectPopupMenu->Append(NODE_POPUP_OUTPORT_BASE + i,
            node->getOutputPort(i)->getName());
        if (_mouseMode == MM_CONNECTING_OUTPUT) {
            portMenu->Enable(NODE_POPUP_OUTPORT_BASE + i, false);
            _connectPopupMenu->Enable(NODE_POPUP_OUTPORT_BASE + i, false);
        }
    }
}

// MainWindow Implementation ---------------------------------------------------

MainWindow::MainWindow(wxWindow* parent, wxWindowID id, const wxString& title,
    const wxPoint& pos, const wxSize& size, long style) :
    wxFrame(parent, id, title, pos, size, style),
    _statusBar(NULL), _splitterWnd(NULL), _activeCanvas(NULL), _sessionLog(NULL)
{
    SetIcon(wxICON(darwin));

    wxMenu *file_menu = new wxMenu;
    wxMenu *file_export_menu = new wxMenu;
    wxMenu *edit_menu = new wxMenu;
    wxMenu *network_menu = new wxMenu;
    wxMenu *database_menu = new wxMenu;
    wxMenu *options_menu = new wxMenu;
    _windowMenu = new wxMenu;
    wxMenu *help_menu = new wxMenu;

    file_menu->Append(FILE_NEW, "&New\tCtrl-N", "New dataflow network");
    file_menu->Append(FILE_OPEN, "&Open...\tCtrl-O", "Open dataflow network");
    file_menu->AppendSeparator();
    file_menu->Append(FILE_SAVE, "&Save\tCtrl-S", "Save dataflow network");
    file_menu->Append(FILE_SAVEAS, "Save &As...", "Save dataflow network");
    file_menu->AppendSeparator();
    file_export_menu->Append(FILE_EXPORT_HTML, "Export &HTML...", "Export HTML report");
    file_export_menu->Append(FILE_EXPORT_CODE, "Export &Code...", "Export source code");
    file_export_menu->Append(FILE_EXPORT_SCRIPT, "Export &Script...", "Export network generation script");
    file_menu->AppendSubMenu(file_export_menu, "&Export");
    file_menu->AppendSeparator();
    file_menu->Append(FILE_CLOSE, "&Close", "Close dataflow network");
    file_menu->Append(FILE_CLOSEALL, "Close All", "Close all dataflow networks");
    file_menu->AppendSeparator();
    file_menu->Append(FILE_EXIT, "E&xit\tAlt-X", "Exit this program");

    edit_menu->Append(EDIT_UNDO, "&Undo\tCtrl-Z", "Undo last change");
    edit_menu->Append(EDIT_REDO, "&Redo\tCtrl-Y", "Redo last change");
    edit_menu->AppendSeparator();
    edit_menu->Append(EDIT_CUT, "&Cut\tCtrl-X", "Cut selected nodes");
    edit_menu->Append(EDIT_COPY, "Copy\tCtrl-C", "Copy selected nodes");
    edit_menu->Append(EDIT_PASTE, "&Paste\tCtrl-V", "Paste copied nodes");
    edit_menu->Append(EDIT_DELETE, "&Delete\tDEL", "Delete selected nodes");
    edit_menu->AppendSeparator();
    edit_menu->Append(EDIT_FIND, "&Find...", "Find nodes");
    edit_menu->AppendSeparator();
    edit_menu->Append(EDIT_SELECTALL, "Select &all\tCtrl-A", "Select all nodes");
    edit_menu->Append(EDIT_DESELECTALL, "Deselect all", "Deselect all nodes");

    network_menu->Append(NETWORK_RESET, "&Reset", "Reset all network parameters and clear data");
    network_menu->Append(NETWORK_EVALUATE, "&Evaluate\tF9", "Evaluate network forwards");
    network_menu->Append(NETWORK_UPDATE, "&Update", "Evaluate network forwards (unprocessed records only)");
    network_menu->Append(NETWORK_INITPARAMS, "&Initialize\tF5", "Initialize network parameters (updating forwards where necessary)");
    network_menu->AppendSeparator();
    network_menu->AppendCheckItem(NETWORK_GRIDSNAP, "Snap to &Grid", "Snap nodes to grid when inserting and moving");

    database_menu->Append(DATABASE_CONNECT, "&Connect...", "Connect to data storage");
    database_menu->Append(DATABASE_VIEW_TABLES, "&View Table...", "View database tables");
    database_menu->Append(DATABASE_VIEW_INSTANCES, "View &Instance...", "View data instances");
    database_menu->AppendSeparator();
    database_menu->Append(DATABASE_IMPORT_COLOURS, "Import Colo&urs...", "Import data partitioning (colours)");
    database_menu->Append(DATABASE_RANDOMIZE_COLOURS, "&Randomize Colours...", "Randomly sample data partitioning (colours)");
    database_menu->AppendSeparator();
    database_menu->Append(DATABASE_FLUSH_CACHE, "&Flush Cache", "Flush and clear the record cache");

    options_menu->AppendRadioItem(OPTIONS_DISPLAY_VERBOSE, "Display &Audit Messages", "Display audit (verbose) messages");
    options_menu->AppendRadioItem(OPTIONS_DISPLAY_MESSAGE, "Display &Messages", "Display messages, warnings and errors");
    options_menu->AppendRadioItem(OPTIONS_DISPLAY_WARNING, "Display &Warnings", "Display warnings and errors only");
    options_menu->AppendSeparator();
    options_menu->AppendCheckItem(OPTIONS_BEEP, "&Beep", "Beep on completion");
    options_menu->AppendSeparator();
    options_menu->Append(OPTIONS_CLEAR_LOG, "&Clear log...", "Clear session log");
    options_menu->Append(OPTIONS_SAVE_LOG, "&Save log...", "Save session log");

    help_menu->Append(HELP_DRWN_CONTENTS, "&Contents...\tF1", "Show help information");
    help_menu->Append(HELP_RELEASE_NOTES, "&Release Notes...", "Show release notes");
    help_menu->Append(HELP_ABOUT, "&About...", "Show about dialog");

    // add all submenus
    wxMenuBar *menu_bar = new wxMenuBar();
    menu_bar->Append(file_menu, "&File");
    menu_bar->Append(edit_menu, "&Edit");
    menu_bar->Append(network_menu, "&Network");
    menu_bar->Append(database_menu, "&Database");
    menu_bar->Append(options_menu, "&Options");
    menu_bar->Append(_windowMenu, "&Window");
    menu_bar->Append(help_menu, "&Help");
    SetMenuBar(menu_bar);

    // set default options
    network_menu->Check(NETWORK_GRIDSNAP, true);
    options_menu->Check(OPTIONS_BEEP, false);
    options_menu->Check(OPTIONS_DISPLAY_VERBOSE, drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE);
    options_menu->Check(OPTIONS_DISPLAY_MESSAGE, drwnLogger::getLogLevel() == DRWN_LL_MESSAGE);
    options_menu->Check(OPTIONS_DISPLAY_WARNING, drwnLogger::getLogLevel() <= DRWN_LL_WARNING);

    // setup statusbar
    _statusBar = new drwnStatusBar(this);
    this->SetStatusBar(_statusBar);
    this->SetStatusBarPane(3);

    // setup subwindows
#if 0
    _splitterWnd = new wxBoxSizer(wxVERTICAL);
    _splitterWnd->SetMinSize(wxSize(320, 240));
    _splitterWnd->Add(_activeCanvas = new MainCanvas(this), 1, wxEXPAND | wxALL);
    _splitterWnd->Add(_sessionLog = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
            wxTE_MULTILINE | wxTE_PROCESS_ENTER | wxRESIZE_BORDER | wxVSCROLL | wxTE_RICH), 0, wxEXPAND | wxALL);
    SetSizer(_splitterWnd);
#else
    _splitterWnd = new wxSplitterWindow(this, wxID_ANY);
    _splitterWnd->SetMinimumPaneSize(32);
    _activeCanvas = new MainCanvas(_splitterWnd);
    _sessionLog = new wxTextCtrl(_splitterWnd, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
        wxTE_MULTILINE | wxTE_PROCESS_ENTER | wxRESIZE_BORDER | wxVSCROLL | wxTE_RICH);
    _splitterWnd->SplitHorizontally(_activeCanvas, _sessionLog, -96);
    _splitterWnd->SetSashGravity(1.0);
    Center();
#endif
    _sessionLog->SetBackgroundColour(wxColour(255, 255, 192));
    _activeCanvas->SetFocus();
    _canvases.push_back(_activeCanvas);
    updateGUIElements();

    // update session log with current date and time
    _sessionLog->SetDefaultStyle(wxTextAttr(*wxBLACK));
    wxDateTime now = wxDateTime::Now();
    _sessionLog->AppendText(wxString("Session log started: ") + now.Format() + wxString("\n"));
}

MainWindow::~MainWindow()
{
    SetStatusBar(NULL);
    if (_statusBar != NULL)
        delete _statusBar;

    clearCopyBuffer();

    // delete canvases (active canvas will be destroyed with window)
    for (int i = 0; i < (int)_canvases.size(); i++) {
        if (_canvases[i] != _activeCanvas) {
            delete _canvases[i];
        }
    }

    if (drwnCodeProfiler::enabled) {
        drwnCodeProfiler::print();
    }
}

void MainWindow::on_idle(wxIdleEvent& event)
{
    // flush some data records
    drwnDataCache::get().idleFlush();
}

void MainWindow::on_close(wxCloseEvent& event)
{
    // not implemented yet
    event.Skip();
}

void MainWindow::on_file_menu(wxCommandEvent& event)
{
    if (event.GetId() == FILE_NEW) {
        _canvases.push_back(new MainCanvas(_splitterWnd));
        _splitterWnd->ReplaceWindow(_activeCanvas, _canvases.back());
        _activeCanvas = _canvases.back();
        updateGUIElements();

    } else if (event.GetId() == FILE_OPEN) {
        wxFileDialog dlg(this, "Open data flow graph",
            "", "", "XML Files (*.xml)|*.xml|All Files (*.*)|*.*",
            wxFD_OPEN | wxFD_CHANGE_DIR);
        if (dlg.ShowModal() == wxID_OK) {
            openGraphFile(dlg.GetPath().c_str());
        }

    } else if ((event.GetId() == FILE_SAVE) && !_activeCanvas->_filename.empty()) {
        _activeCanvas->_graph->write(_activeCanvas->_filename.c_str());
        // TODO: clear dirty flag
    } else if ((event.GetId() == FILE_SAVEAS) || (event.GetId() == FILE_SAVE)) {
        drwnGraph *graph = _activeCanvas->_graph;
        wxFileDialog dlg(this, string("Save ") + graph->getTitle() + string(" as:"),
            "", _activeCanvas->_filename, "XML Files (*.xml)|*.xml|All Files (*.*)|*.*",
            wxFD_SAVE | wxFD_CHANGE_DIR);
        if (dlg.ShowModal() == wxID_OK) {
            _activeCanvas->_filename = dlg.GetPath();
            graph->write(dlg.GetPath().c_str());
            // TODO: clear dirty flag
        }

    } else if (event.GetId() == FILE_EXPORT_HTML) {
        drwnGraph *graph = _activeCanvas->_graph;
        string filenameProposal;
        if (!_activeCanvas->_filename.empty()) {
            filenameProposal = drwn::strReplaceExt(_activeCanvas->_filename, string(".html"));
        }
        wxFileDialog dlg(this, string("Export HTML for ") + graph->getTitle() + string(" as:"),
            "", filenameProposal, "HTML Files (*.html)|*.html|All Files (*.*)|*.*",
            wxFD_SAVE | wxFD_CHANGE_DIR);
        if (dlg.ShowModal() == wxID_OK) {
            drwnHTMLReport report;
            report.write(dlg.GetPath().c_str(), graph);
        }

    } else if (event.GetId() == FILE_EXPORT_HTML) {
        NOT_IMPLEMENTED_YET;

    } else if (event.GetId() == FILE_EXPORT_SCRIPT) {
        NOT_IMPLEMENTED_YET;

    } else if (event.GetId() == FILE_CLOSE) {
        // TODO: ask to save if dirty
        if (_canvases.size() == 1) {
            _canvases[0] = new MainCanvas(_splitterWnd);
        } else {
            for (vector<MainCanvas *>::iterator it = _canvases.begin(); it != _canvases.end(); it++) {
                if (*it == _activeCanvas) {
                    _canvases.erase(it);
                    break;
                }
            }
        }
        _splitterWnd->ReplaceWindow(_activeCanvas, _canvases.back());
        delete _activeCanvas;
        _activeCanvas = _canvases.back();
        updateGUIElements();

    } else if (event.GetId() == FILE_CLOSEALL) {
        // TODO: ask to save if dirty
        for (vector<MainCanvas *>::iterator it = _canvases.begin(); it != _canvases.end(); it++) {
            if (*it != _activeCanvas) {
                delete *it;
            }
        }
        _canvases.clear();
        _canvases.push_back(new MainCanvas(_splitterWnd));
        _splitterWnd->ReplaceWindow(_activeCanvas, _canvases.back());
        delete _activeCanvas;
        _activeCanvas = _canvases.back();
        updateGUIElements();

    } else if (event.GetId() == FILE_EXIT) {
        Close(true);
    }

    Refresh(false);
    Update();
}

void MainWindow::on_edit_menu(wxCommandEvent& event)
{
    if (event.GetId() == EDIT_CUT) {
        clearCopyBuffer();
        _copyBuffer = _activeCanvas->_graph->copySubGraph(_activeCanvas->_selectedNodes);
        for (set<drwnNode *>::iterator it = _activeCanvas->_selectedNodes.begin();
             it != _activeCanvas->_selectedNodes.end(); it++) {
            _activeCanvas->_graph->delNode(*it);
        }
        _activeCanvas->_selectedNodes.clear();
        _activeCanvas->_activeNode = NULL;
        DRWN_LOG_VERBOSE(_copyBuffer.size() << " nodes cut");
    } else if (event.GetId() == EDIT_COPY) {
        clearCopyBuffer();
        _copyBuffer = _activeCanvas->_graph->copySubGraph(_activeCanvas->_selectedNodes);
        DRWN_LOG_VERBOSE(_copyBuffer.size() << " nodes copied");
    } else if (event.GetId() == EDIT_PASTE) {
        int width, height, x, y;
        _activeCanvas->GetClientSize(&width, &height);
        _activeCanvas->CalcUnscrolledPosition(width / 2, height / 2, &x, &y);
        if (_activeCanvas->_bSnapToGrid) {
            x = 32 * ((int)(x + 16) / 32);
            y = 32 * ((int)(y + 16) / 32);
        }
        _activeCanvas->_graph->pasteSubGraph(_copyBuffer, x, y);
        _activeCanvas->_selectedNodes.clear();
        for (int i = _activeCanvas->_graph->numNodes() - _copyBuffer.size();
             i < _activeCanvas->_graph->numNodes(); i++) {
            _activeCanvas->_selectedNodes.insert(_activeCanvas->_graph->getNode(i));
        }
        DRWN_LOG_VERBOSE(_copyBuffer.size() << " nodes pasted");
    } else if (event.GetId() == EDIT_DELETE) {
        if (_activeCanvas->_selectedNodes.empty()) {
            wxMessageBox("No nodes selected.", "Delete", wxOK | wxICON_INFORMATION, this);
        } else {
            wxMessageDialog dlg(this,
                wxString::Format("Delete %d nodes? All associated data will be lost.",
                    (int)_activeCanvas->_selectedNodes.size()), "Delete", wxYES_NO);
            if (dlg.ShowModal() == wxID_YES) {
                for (set<drwnNode *>::iterator it = _activeCanvas->_selectedNodes.begin();
                     it != _activeCanvas->_selectedNodes.end(); it++) {
                    _activeCanvas->_graph->delNode(*it);
                }
                _activeCanvas->_selectedNodes.clear();
                _activeCanvas->_activeNode = NULL;
            }
        }
    } else if (event.GetId() == EDIT_FIND) {
	wxTextEntryDialog dlg(this, "Search string:", "Find");
        if (dlg.ShowModal() == wxID_OK) {
            string searchStr = string(dlg.GetValue().c_str());
            const drwnGraph *g = _activeCanvas->_graph;
            _activeCanvas->_selectedNodes.clear();
            for (int i = 0; i < g->numNodes(); i++) {
                if (g->getNode(i)->getName().find(searchStr) != string::npos) {
                    _activeCanvas->_selectedNodes.insert(g->getNode(i));
                }
            }
        }
    } else if (event.GetId() == EDIT_SELECTALL) {
        _activeCanvas->selectAllNodes();
    } else if (event.GetId() == EDIT_DESELECTALL) {
        _activeCanvas->deselectAllNodes();
    }

    Refresh(false);
    Update();
}

void MainWindow::on_network_menu(wxCommandEvent& event)
{
    if (event.GetId() == NETWORK_RESET) {
        _activeCanvas->_graph->resetParameters(_activeCanvas->_selectedNodes);
    } else if (event.GetId() == NETWORK_EVALUATE) {
        _activeCanvas->_graph->evaluateForwards(_activeCanvas->_selectedNodes);
    } else if (event.GetId() == NETWORK_UPDATE) {
        _activeCanvas->_graph->updateForwards(_activeCanvas->_selectedNodes);
    } else if (event.GetId() == NETWORK_INITPARAMS) {
        _activeCanvas->_graph->initializeParameters(_activeCanvas->_selectedNodes);
    } else if (event.GetId() == NETWORK_GRIDSNAP) {
        _activeCanvas->snapToGrid() = event.IsChecked();
    }

    Refresh(false);
    Update();
}

void MainWindow::on_database_menu(wxCommandEvent& event)
{
    if (event.GetId() == DATABASE_CONNECT) {
        drwnDatabase *db = _activeCanvas->_graph->getDatabase();
        wxDirDialog dlg(this, string("Select database for ") + _activeCanvas->_graph->getTitle(),
            (db->isPersistent() ? drwn::strDirectory(db->name()).c_str() : ""), wxDD_DEFAULT_STYLE);
        if (dlg.ShowModal() == wxID_OK) {
            _activeCanvas->_graph->setDatabase(dlg.GetPath().c_str());
        }
        updateGUIElements();

    } else if (event.GetId() == DATABASE_VIEW_TABLES) {
        drwnDatabase *db = _activeCanvas->_graph->getDatabase();
        vector<string> tblNames = db->getTableNames();
        if (tblNames.empty()) {
            DRWN_LOG_ERROR("database has no tables");
            return;
        }

        wxString *choices = new wxString[tblNames.size()];
        for (unsigned i = 0; i < tblNames.size(); i++) {
            choices[i] = wxString::Format("%s (%d records)", tblNames[i].c_str(),
                db->getTable(tblNames[i])->numRecords());
        }

        wxSingleChoiceDialog dlg(this, "Select table:", "View Table",
            (int)tblNames.size(), choices);
        if (dlg.ShowModal() == wxID_OK) {
            DRWN_LOG_MESSAGE("TODO: show table " << tblNames[dlg.GetSelection()]);
            NOT_IMPLEMENTED_YET;
        }
        delete[] choices;

    } else if (event.GetId() == DATABASE_VIEW_INSTANCES) {
        drwnDatabase *db = _activeCanvas->_graph->getDatabase();
        vector<string> keys = db->getAllKeys();
        if (keys.empty()) {
            DRWN_LOG_ERROR("database has no records");
            return;
        }

        wxString *choices = new wxString[keys.size()];
        for (unsigned i = 0; i < keys.size(); i++) {
            choices[i] = wxString::Format("%s (colour %d)", keys[i].c_str(),
                db->getColour(keys[i]));
        }

        wxSingleChoiceDialog dlg(this, "Select data instance:", "View Records",
            (int)keys.size(), choices);
        if (dlg.ShowModal() == wxID_OK) {
            string k = keys[dlg.GetSelection()];
            drwnTextEditor recordDlg(this, k.c_str(), true);

            vector<string> tblNames = db->getTableNames();
            for (vector<string>::const_iterator it = tblNames.begin(); it != tblNames.end(); it++) {
                drwnDataTable *tbl = db->getTable(*it);
                recordDlg.addLine(it->c_str(), true);
                if (tbl->hasKey(k)) {
                    drwnDataRecord *rec = tbl->lockRecord(k);
                    if (rec->isEmpty()) {
                        recordDlg.addLine("  empty");
                    } else {
                        const MatrixXd& d = rec->data();
                        for (int i = 0; i < d.rows(); i++) {
                            vector<double> v(d.cols());
                            Eigen::Map<VectorXd>(&v[0], v.size()) = d.row(i);
                            recordDlg.addLine((string("  ") + toString(v)).c_str());
                        }
                    }
                    tbl->unlockRecord(k);
                } else {
                    recordDlg.addLine("  no data");
                }
                recordDlg.addLine();
            }
            recordDlg.ShowModal();
        }
        delete[] choices;

    } else if (event.GetId() == DATABASE_IMPORT_COLOURS) {
        NOT_IMPLEMENTED_YET;

    } else if (event.GetId() == DATABASE_RANDOMIZE_COLOURS) {
        wxTextEntryDialog dlg(this, "Enter number of data partitions (colours):",
            "Randomize Colours", "2");
        if (dlg.ShowModal()) {
            drwnDatabase *db = _activeCanvas->_graph->getDatabase();
            int numColours = atoi(dlg.GetValue().c_str());
            if (numColours <= 0) {
                db->clearColours();
            } else {
                // initialize random number generator
                drwnInitializeRand();

                // randomly assign colours
                vector<string> keys = db->getAllKeys();
                for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
                    db->setColour(*it, rand() % numColours);
                }
            }
        }

    } else if (event.GetId() == DATABASE_FLUSH_CACHE) {
        drwnDataCache::get().flush();
        drwnDataCache::get().clear();
    }

    Refresh(false);
    Update();
}

void MainWindow::on_options_menu(wxCommandEvent& event)
{
    if (event.GetId() == OPTIONS_DISPLAY_VERBOSE) {
        drwnLogger::setLogLevel(DRWN_LL_VERBOSE);
    } else if (event.GetId() == OPTIONS_DISPLAY_MESSAGE) {
        drwnLogger::setLogLevel(DRWN_LL_MESSAGE);
    } else if (event.GetId() == OPTIONS_DISPLAY_WARNING) {
        drwnLogger::setLogLevel(DRWN_LL_WARNING);
    } else if (event.GetId() == OPTIONS_BEEP) {
        // TODO
        ::wxBell();
    } else if (event.GetId() == OPTIONS_CLEAR_LOG) {
        wxMessageDialog dlg(this, "Do you really want to clear all session log messages?",
            "Session Log", wxYES_NO | wxICON_QUESTION);
        if (dlg.ShowModal() == wxID_YES) {
            _sessionLog->Clear();

            // update session log with current date and time
            _sessionLog->SetDefaultStyle(wxTextAttr(*wxBLACK));
            wxDateTime now = wxDateTime::Now();
            _sessionLog->AppendText(wxString("Session log restarted: ") + now.Format() + wxString("\n"));
        }
    } else if (event.GetId() == OPTIONS_SAVE_LOG) {
        wxFileDialog dlg(this, "Save log file as:", "", "",
            "Text Files (*.txt)|*.txt|All Files (*.*)|*.*", wxFD_SAVE);
        if (dlg.ShowModal() == wxID_OK) {
            // TODO: save current messages too
            drwnLogger::initialize(dlg.GetPath().c_str(), false);
        }
    }

    Refresh(false);
    Update();
}

void MainWindow::on_window_menu(wxCommandEvent& event)
{
    if ((event.GetId() >= WINDOW_MENU_BASE) &&
        (event.GetId() < WINDOW_MENU_BASE + 100)) {
        int windowId = event.GetId() - WINDOW_MENU_BASE;
        if (_canvases[windowId] != _activeCanvas) {
            _splitterWnd->ReplaceWindow(_activeCanvas, _canvases[windowId]);
            _activeCanvas = _canvases[windowId];
        }
        updateGUIElements();
    }

    Refresh(false);
    Update();
}

void MainWindow::on_help_menu(wxCommandEvent& event)
{
    if (event.GetId() == HELP_DRWN_CONTENTS) {
        ::wxLaunchDefaultBrowser("http://drwn.anu.edu.au/", wxBROWSER_NEW_WINDOW);
        DRWN_LOG_ERROR("Help|Contents not implemented yet");
    } else if (event.GetId() == HELP_RELEASE_NOTES) {
        ::wxLaunchDefaultBrowser("http://drwn.anu.edu.au/", wxBROWSER_NEW_WINDOW);
        DRWN_LOG_ERROR("Help|Release Notes not implemented yet");
    } else if (event.GetId() == HELP_ABOUT) {
        wxAboutDialogInfo info;
        info.SetName("Darwin");
        info.SetVersion(DRWN_VERSION);
        info.SetDescription("A framework for machine learning\nresearch and development.");
        info.SetCopyright(DRWN_COPYRIGHT);

        wxAboutBox(info);
    }
}

// status
void MainWindow::logMessage(const char *msg, const wxTextAttr style)
{
    _sessionLog->SetDefaultStyle(style);
    _sessionLog->AppendText(wxString(msg) + wxString("\n"));
}

void MainWindow::updateProgress(const char *status, double progress)
{
    static clock_t _lastClock = clock();
    _statusBar->updateMessage(wxString(status));
    _statusBar->updateProgress(progress);
    _statusBar->Refresh(false);
    //_statusBar->Update(); // TODO: change to timer update
    //_sessionLog->Refresh(false);
    //_sessionLog->Update();

    // throttle updates
    clock_t thisClock = clock();
    if (thisClock - _lastClock > CLOCKS_PER_SEC / 100) {
        this->Update();
        //::wxSafeYield(NULL, true);
        _lastClock = thisClock;
    }

    // check for abort
    if (::wxGetKeyState(WXK_ESCAPE)) {
        if (drwnLogger::isRunning()) {
            DRWN_LOG_WARNING("Terminating current operation. Please wait...");
            drwnLogger::setRunning(false);
            ::wxSafeYield(NULL, true);
        }
    }
}

void MainWindow::updateGUIElements()
{
    // hide non-active windows
    for (int i = 0; i < (int)_canvases.size(); i++) {
        _canvases[i]->Show(_canvases[i] == _activeCanvas);
    }

    // update window menu
    // TODO: hack?
    while (_windowMenu->GetMenuItemCount() > 0) {
        _windowMenu->Destroy(_windowMenu->FindItemByPosition(0));
    }

    for (int i = 0; i < (int)_canvases.size(); i++) {
        _windowMenu->Append(WINDOW_MENU_BASE + i,
            _canvases[i]->_graph->getTitle());
        if (_canvases[i] == _activeCanvas) {
            _windowMenu->Enable(WINDOW_MENU_BASE + i, false);
        }
    }

    // TODO: make Window menu hidden if only one window

    // update status bar
    _statusBar->updateDatabase(_activeCanvas->_graph->getDatabase());
}

void MainWindow::clearCopyBuffer()
{
    for (set<drwnNode *>::iterator it = _copyBuffer.begin(); it != _copyBuffer.end(); it++) {
        delete *it;
    }
    _copyBuffer.clear();
}

bool MainWindow::openGraphFile(const char *filename)
{
    DRWN_ASSERT(filename != NULL);

    // check if file is already open
    for (vector<MainCanvas *>::const_iterator it = _canvases.begin();
         it != _canvases.end(); it++) {
        if ((*it)->_filename == string(filename)) {

            wxMessageBox(wxString::Format("Dataflow graph \"%s\" is already open.", filename),
                "Open...", wxOK | wxICON_EXCLAMATION, NULL);

            if (*it != _activeCanvas) {
                _splitterWnd->ReplaceWindow(_activeCanvas, *it);
                _activeCanvas = *it;
                updateGUIElements();
            }

            return true;
        }
    }

    // otherwise open new file
    if (_activeCanvas->_graph->numNodes() != 0) {
        _canvases.push_back(new MainCanvas(_splitterWnd));
        _splitterWnd->ReplaceWindow(_activeCanvas, _canvases.back());
        _activeCanvas = _canvases.back();
    }
    _activeCanvas->_graph->read(filename);
    _activeCanvas->_filename = string(filename);
    _activeCanvas->Scroll(0, 0);
    updateGUIElements();

    return true;
}

// Darwin Implementation ------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << "\n";
    cerr << "USAGE: ./darwin [OPTIONS] (<network>)\n";
    cerr << "OPTIONS:\n"
         << "  -plugin <path>    :: load plugin\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

bool DarwinApp::OnInit()
{
#ifdef __WXMAC__
    // application registration (Mac OS X)
    ProcessSerialNumber PSN;
    GetCurrentProcess(&PSN);
    TransformProcessType(&PSN, kProcessTransformToForegroundApplication);
#endif

    // parse command-line options
    int argc = wxAppConsole::argc;
    char **argv = wxAppConsole::argv;
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_OPTION_BEGIN("-plugin", p)
#if defined(__LINUX__)
            void *h = dlopen(p[0], RTLD_NOW);
            if (h == NULL) {
                DRWN_LOG_ERROR(dlerror());
            }
#endif
        DRWN_CMDLINE_OPTION_END(1)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC > 1) {
        usage();
        return false;
    }

    // setup main window
    gMainWindow = new MainWindow(NULL, wxID_ANY, "Darwin GUI",
        wxDefaultPosition, wxSize(640, 480));
    SetTopWindow(gMainWindow);

    gMainWindow->Show();
    gMainWindow->SetFocus();

    // initialize drwnLogger callbacks
    drwnLogger::showMessageCallback = messageCallback;
    drwnLogger::showWarningCallback = warningCallback;
    drwnLogger::showErrorCallback = errorCallback;
    drwnLogger::showFatalCallback = fatalCallback;
    drwnLogger::showProgressCallback = progressCallback;

    // load network
    if (DRWN_CMDLINE_ARGC == 1) {
        string fullPath = string(DRWN_CMDLINE_ARGV[0]);
        drwnChangeCurrentDir(drwn::strDirectory(fullPath).c_str());
        gMainWindow->openGraphFile(drwn::strFilename(fullPath).c_str());
    }

    return true;
}

int DarwinApp::OnExit()
{
    // reset drwnLogger callbacks
    drwnLogger::showMessageCallback = NULL;
    drwnLogger::showWarningCallback = NULL;
    drwnLogger::showErrorCallback = NULL;
    drwnLogger::showFatalCallback = NULL;
    drwnLogger::showProgressCallback = NULL;

    return 0;
}

IMPLEMENT_APP(DarwinApp)

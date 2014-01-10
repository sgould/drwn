/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    darwin.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Darwin GUI for creating and evaluating data flow algorithms.
**
*****************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <map>

#include "wx/wx.h"
#include "wx/glcanvas.h"
#include "wx/utils.h"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnNodes.h"

#include "drwnStatusBar.h"

// wxWidgets Event Constants --------------------------------------------------

// MainCanvas messages
enum {
    // node popup
    NODE_POPUP_SET_NAME = wxID_HIGHEST + 100,
    NODE_POPUP_PROPERTIES = wxID_HIGHEST + 110,
    NODE_POPUP_SHOWHIDE = wxID_HIGHEST + 120,
    NODE_POPUP_EVALUATE = wxID_HIGHEST + 200,
    NODE_POPUP_UPDATE = wxID_HIGHEST + 210,
    NODE_POPUP_PROPBACK = wxID_HIGHEST + 220,
    NODE_POPUP_RESETPARAMS = wxID_HIGHEST + 230,
    NODE_POPUP_INITPARAMS = wxID_HIGHEST + 240,
    NODE_POPUP_INPORT_BASE = wxID_HIGHEST + 400,
    NODE_POPUP_OUTPORT_BASE = wxID_HIGHEST + 500,
    NODE_POPUP_DISCONNECT = wxID_HIGHEST + 600,
    NODE_POPUP_DELETE = wxID_HIGHEST + 300,

    // network popup
    POPUP_SET_TITLE = wxID_HIGHEST + 800,
    POPUP_NODE_INSERT_BASE = wxID_HIGHEST + 1000
};

// MainWindow messages
enum
{
    FILE_NEW = wxID_NEW,
    FILE_OPEN = wxID_OPEN,
    FILE_SAVE = wxID_SAVE,
    FILE_SAVEAS = wxID_SAVEAS,
    FILE_CLOSE = wxID_CLOSE,
    FILE_CLOSEALL = wxID_HIGHEST + 21,
    FILE_EXPORT = wxID_HIGHEST + 30,
    FILE_EXPORT_HTML = wxID_HIGHEST + 31,
    FILE_EXPORT_CODE = wxID_HIGHEST + 32,
    FILE_EXPORT_SCRIPT = wxID_HIGHEST + 33,
    FILE_EXIT = wxID_EXIT,

    EDIT_UNDO = wxID_UNDO,
    EDIT_REDO = wxID_REDO,
    EDIT_CUT = wxID_CUT,
    EDIT_COPY = wxID_COPY,
    EDIT_PASTE = wxID_PASTE,
    EDIT_DELETE = wxID_DELETE,
    EDIT_FIND = wxID_FIND,
    EDIT_SELECTALL = wxID_SELECTALL,
    EDIT_DESELECTALL = wxID_HIGHEST + 131,

    NETWORK_RESET = wxID_HIGHEST + 200,
    NETWORK_EVALUATE = wxID_HIGHEST + 210,
    NETWORK_UPDATE = wxID_HIGHEST + 220,
    NETWORK_BACKPROP = wxID_HIGHEST + 230,
    NETWORK_INITPARAMS = wxID_HIGHEST + 240,
    NETWORK_GRIDSNAP = wxID_HIGHEST + 260,

    DATABASE_CONNECT = wxID_HIGHEST + 300,
    DATABASE_VIEW_TABLES = wxID_HIGHEST + 301,
    DATABASE_VIEW_INSTANCES = wxID_HIGHEST + 302,
    DATABASE_IMPORT_COLOURS = wxID_HIGHEST + 310,
    DATABASE_RANDOMIZE_COLOURS = wxID_HIGHEST + 312,
    DATABASE_FLUSH_CACHE = wxID_HIGHEST + 350,

    OPTIONS_DISPLAY_VERBOSE = wxID_HIGHEST + 400,
    OPTIONS_DISPLAY_MESSAGE = wxID_HIGHEST + 401,
    OPTIONS_DISPLAY_WARNING = wxID_HIGHEST + 402,
    OPTIONS_BEEP = wxID_HIGHEST + 410,
    OPTIONS_CLEAR_LOG = wxID_HIGHEST + 420,
    OPTIONS_SAVE_LOG = wxID_HIGHEST + 421,
    OPTIONS_TILE_WINDOWS = wxID_HIGHEST + 430,

    WINDOW_MENU_BASE = wxID_HIGHEST + 500,

    HELP_DRWN_CONTENTS = wxID_HELP_CONTENTS,
    HELP_RELEASE_NOTES = wxID_HIGHEST + 810,
    HELP_ABOUT = wxID_ABOUT
};

// Mouse Modes ----------------------------------------------------------------

typedef enum {
    MM_NONE, MM_DRAGGING, MM_SELECTING, MM_SELECTING_NOERASE,
    MM_CONNECTING_INPUT, MM_CONNECTING_OUTPUT
} TMouseMode;

// MainCanvas Class -----------------------------------------------------------
// This is required under Linux because wxFrame doesn't get keyboard focus
// without a child window.

class MainCanvas : public wxScrolledWindow
{
    friend class MainWindow;

 public:
    MainCanvas(wxWindow *parent,
        wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize,
        long style = wxDEFAULT_FRAME_STYLE | wxSUNKEN_BORDER | wxWANTS_CHARS,
        const wxString& name = wxPanelNameStr);
    ~MainCanvas();

    // event handling
    void on_erase_background(wxEraseEvent &event);
    void on_paint(wxPaintEvent &event);
    void on_size(wxSizeEvent &event);
    void on_key(wxKeyEvent &event);
    void on_mouse(wxMouseEvent &event);
    void on_popup_menu(wxCommandEvent &event);

    // setting and getting options
    const bool& snapToGrid() const { return _bSnapToGrid; }
    bool& snapToGrid() { Refresh(false); return _bSnapToGrid; }

    void selectNodesInRegion(int x, int y, int w, int h);
    void selectAllNodes();
    void deselectAllNodes();

 protected:
    void undoableAction();

    void updateImageBuffer();
    void updateStatusBar();

    int findNodeAtLocation(int x, int y) const;
    void updatePortSubMenu(const drwnNode *node);

 protected:    
    // algorithm state
    string _filename;
    drwnGraph *_graph;
    drwnNode *_activeNode;
    drwnDataPort *_activePort;
    set<drwnNode *> _selectedNodes;

    // mouse management
    bool _bSnapToGrid;
    TMouseMode _mouseMode;
    wxPoint _mousePoint;
    wxPoint _lastMousePoint;
    wxPoint _buttonDownPoint;

    // gui elements/state
    static int _creationCount;
    wxMenu *_nodePopupMenu;
    wxMenuItem *_portSubMenuItem;
    wxMenu *_connectPopupMenu;
    wxMenu *_newNodePopupMenu;

    DECLARE_EVENT_TABLE()
};

// MainWindow Class -----------------------------------------------------------

class DarwinApp;

class MainWindow : public wxFrame
{
    friend class MainCanvas;
    friend class DarwinApp;

 public:
    MainWindow(wxWindow* parent,
        wxWindowID id,
        const wxString& title,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize,
        long style = wxDEFAULT_FRAME_STYLE | wxSUNKEN_BORDER | wxWANTS_CHARS);
    ~MainWindow();

    // event callbacks
    void on_idle(wxIdleEvent& event);
    void on_close(wxCloseEvent& event);
    void on_file_menu(wxCommandEvent& event);
    void on_edit_menu(wxCommandEvent& event);
    void on_network_menu(wxCommandEvent& event);
    void on_database_menu(wxCommandEvent& event);
    void on_options_menu(wxCommandEvent& event);
    void on_window_menu(wxCommandEvent& event);
    void on_help_menu(wxCommandEvent& event);

    // status
    void logMessage(const char *msg, const wxTextAttr style = wxTextAttr(*wxBLACK));
    void updateProgress(const char *status, double progress);

 protected:
    void updateGUIElements();
    bool openGraphFile(const char *filename);
    void clearCopyBuffer();

 protected:
    drwnStatusBar *_statusBar;
    wxSplitterWindow *_splitterWnd;
    vector<MainCanvas *> _canvases;
    MainCanvas *_activeCanvas;
    wxTextCtrl *_sessionLog;
    wxMenu *_windowMenu;

    set<drwnNode *> _copyBuffer;

    DECLARE_EVENT_TABLE()
};

// Darwin Application ---------------------------------------------------------

class DarwinApp : public wxApp
{
 public:
    bool OnInit();
    int OnExit();
};

// Global Variables -----------------------------------------------------------

extern MainWindow *gMainWindow;



/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStatusBar.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Darwin statusbar class.
**
*****************************************************************************/

#pragma once

#include "wx/wx.h"
#include "wx/utils.h"

#if !wxUSE_STATUSBAR
    #error "You need to set wxUSE_STATUSBAR to 1"
#endif

#include "drwnBase.h"
#include "drwnEngine.h"

// drwnProgressBar -----------------------------------------------------------

class drwnProgressBar : public wxWindow
{
 protected:
    wxBitmap *_bmpProgress;
    int _progressLevel;
    string _statusText;

 public:
    drwnProgressBar(wxWindow *parent,
        wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize,
        long style = wxDEFAULT_FRAME_STYLE | wxNO_BORDER,
        const wxString& name = wxPanelNameStr);
    ~drwnProgressBar();

    // update progress
    void setProgress(double p);
    void setStatusText(const char *s);

    // event handling
    void on_erase_background(wxEraseEvent &event);
    void on_paint(wxPaintEvent &event);
    void on_size(wxSizeEvent &event);

 protected:
    DECLARE_EVENT_TABLE()
};

// drwnStatusBar --------------------------------------------------------------

class drwnStatusBar : public wxStatusBar
{
 protected:
    static const int NUM_FIELDS;
    drwnProgressBar *_progressBar;
    wxStaticBitmap *_bmpDBNone;
    wxStaticBitmap *_bmpDBConnected;

 public:
    drwnStatusBar(wxWindow *parent);
    virtual ~drwnStatusBar();

    // update status bar
    void updateProgress(double p = 0.0);
    void updateMemory(double m = 0.0);
    void updateDatabase(const drwnDatabase *db);
    void updateMessage(const wxString& msg);

    // event handling
    void on_erase_background(wxEraseEvent &event);
    void on_paint(wxPaintEvent &event);
    void on_size(wxSizeEvent& event);

 private:
    DECLARE_EVENT_TABLE()
};

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnStatusBar.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "wx/wxprec.h"
#include "wx/utils.h"
#include "wx/dcbuffer.h"

#include "drwnStatusBar.h"

#include "resources/dbdisconnected.xpm"
#include "resources/db0.xpm"
#include "resources/progressbar.xpm"

// Event Tables --------------------------------------------------------------

BEGIN_EVENT_TABLE(drwnProgressBar, wxWindow)
    EVT_ERASE_BACKGROUND(drwnProgressBar::on_erase_background)
    EVT_PAINT(drwnProgressBar::on_paint)
    EVT_SIZE(drwnProgressBar::on_size)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(drwnStatusBar, wxStatusBar)
    EVT_ERASE_BACKGROUND(drwnStatusBar::on_erase_background)
    //EVT_PAINT(drwnStatusBar::on_paint)
    EVT_SIZE(drwnStatusBar::on_size)
END_EVENT_TABLE()

// drwnProgressBar Implementation --------------------------------------------

drwnProgressBar::drwnProgressBar(wxWindow *parent, wxWindowID id,const wxPoint& pos,
    const wxSize& size, long style, const wxString& name) :
    wxWindow(parent, id, pos, size, style, name), _bmpProgress(NULL), _progressLevel(0)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    _bmpProgress = new wxBitmap(wxBITMAP(progressbar));
}

drwnProgressBar::~drwnProgressBar()
{
    if (_bmpProgress != NULL)
        delete _bmpProgress;
}

void drwnProgressBar::setProgress(double p)
{
    _progressLevel = std::max(0, std::min((int)(p * _bmpProgress->GetWidth()),
            _bmpProgress->GetWidth()));
}

void drwnProgressBar::setStatusText(const char *s)
{
    if (s == NULL) _statusText.clear();
    else _statusText = string(s);
}

void drwnProgressBar::on_erase_background(wxEraseEvent &event)
{
    // do nothing (and avoid flicker)
    event.Skip();
}

void drwnProgressBar::on_paint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.SetBackground(wxBrush(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE)));
    dc.Clear();

    // show progress
    if (_progressLevel != 0) {
        wxSize size = GetClientSize();
        wxBitmap bmpCurrentProgress =
            _bmpProgress->GetSubBitmap(wxRect(0, 0, _progressLevel, _bmpProgress->GetHeight()));
        dc.DrawBitmap(bmpCurrentProgress, 0, (size.y - _bmpProgress->GetHeight()) / 2);
    }

    // draw statusbar text
    if (!_statusText.empty()) {
        dc.SetFont(wxSystemSettings::GetFont(wxSYS_DEFAULT_GUI_FONT));
        //dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE));
        dc.SetTextForeground(wxColor(0, 0, 0));
        dc.DrawText(_statusText.c_str(), 0, 0);
    }

    event.Skip();
}

void drwnProgressBar::on_size(wxSizeEvent& event)
{
    // do nothing
}

// drwnStatusBar Implementation ----------------------------------------------

const int drwnStatusBar::NUM_FIELDS = 4;

drwnStatusBar::drwnStatusBar(wxWindow *parent) : wxStatusBar(parent, wxID_ANY)
{
    static const int widths[NUM_FIELDS] = {150, 100, 36, -1};
    static const int styles[NUM_FIELDS] = {wxSB_NORMAL, wxSB_NORMAL, wxSB_FLAT, wxSB_NORMAL};

    SetBackgroundStyle(wxBG_STYLE_PAINT);
    SetFieldsCount(NUM_FIELDS);
    SetStatusWidths(NUM_FIELDS, widths);
    SetStatusStyles(NUM_FIELDS, styles);

    _progressBar = new drwnProgressBar(this, wxID_ANY);
    _bmpDBNone = new wxStaticBitmap(this, wxID_ANY, wxBITMAP(dbdisconnected));
    _bmpDBConnected = new wxStaticBitmap(this, wxID_ANY, wxBITMAP(db0));
    _bmpDBConnected->Show(false);

    //SetMinHeight(64);
}

drwnStatusBar::~drwnStatusBar()
{
    delete _progressBar;
    delete _bmpDBNone;
    delete _bmpDBConnected;
}

void drwnStatusBar::updateMessage(const wxString& msg)
{
    //wxStatusBar::SetStatusText(msg, 0);
    _progressBar->setStatusText(msg.c_str());
    _progressBar->Refresh(false);
}

void drwnStatusBar::updateProgress(double p)
{
    _progressBar->setProgress(p);
    //_progressBar->Refresh();

    // testing
#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)||(__VISUALC__)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);

    SetStatusText(wxString::Format("Mem: %ldMb (%d%%)",
        (long)(statex.ullTotalPhys - statex.ullAvailPhys) / (1024 * 1024),
        statex.dwMemoryLoad), 1);
#endif
}

void drwnStatusBar::updateDatabase(const drwnDatabase *db)
{
    DRWN_ASSERT(db != NULL);
    bool bConnected = db->isPersistent();
    _bmpDBNone->Show(!bConnected);
    _bmpDBConnected->Show(bConnected);    
}

void drwnStatusBar::on_erase_background(wxEraseEvent &event)
{
    // do nothing (and avoid flicker)
    event.Skip();
}

void drwnStatusBar::on_paint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    event.Skip();
}

void drwnStatusBar::on_size(wxSizeEvent& event)
{
    wxRect rect;

    // reposition progress bar
    GetFieldRect(0, rect);
    rect.x += 1; rect.y += 1; rect.width -= 2; rect.height -= 2;
    _progressBar->SetSize(rect);

    // reposition database icon
    GetFieldRect(2, rect);
    wxSize size = _bmpDBNone->GetSize();
    _bmpDBNone->Move(rect.x, rect.y + (rect.height - size.y) / 2);
    _bmpDBConnected->Move(rect.x, rect.y + (rect.height - size.y) / 2);

    event.Skip();
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextEditor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "wx/wxprec.h"
#include "wx/utils.h"
#include "wx/dialog.h"

#include "drwnBase.h"
#include "drwnTextEditor.h"

using namespace std;

// drwnTextEditor ------------------------------------------------------------

drwnTextEditor::drwnTextEditor(wxWindow *parent, const char *title, bool bReadOnly) :
    wxDialog(parent, wxID_ANY, title, wxDefaultPosition, wxDefaultSize,
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxFRAME_TOOL_WINDOW | wxFULL_REPAINT_ON_RESIZE)
{
    wxBoxSizer *mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->SetMinSize(wxSize(320, 240));

    // create text control
    if (bReadOnly) {
        _text = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition,
            wxDefaultSize, wxEXPAND | wxTE_DONTWRAP | wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH);
    } else {
        _text = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition,
            wxDefaultSize, wxEXPAND | wxTE_DONTWRAP | wxTE_MULTILINE | wxTE_RICH);
    }
    mainSizer->Add(_text, 1, wxEXPAND | wxALL, 10);

    // implement the sizer
    SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);
}

drwnTextEditor::~drwnTextEditor()
{
    // do nothing
}

// i/o
void drwnTextEditor::clear()
{
    _text->Clear();

    Refresh(false);
    Update();
}

void drwnTextEditor::addLine()
{
    _text->AppendText("\n");
}

void drwnTextEditor::addLine(const char *str, bool bBold)
{
    if (bBold) {
        wxFont font = _text->GetDefaultStyle().GetFont();
        font.SetWeight(wxFONTWEIGHT_BOLD);
        _text->SetDefaultStyle(wxTextAttr(*wxBLACK, wxNullColour, font));
        _text->AppendText(str);
        font.SetWeight(wxFONTWEIGHT_NORMAL);
        _text->SetDefaultStyle(wxTextAttr(*wxBLACK, wxNullColour, font));
    } else {
        _text->AppendText(str);
    }
    _text->AppendText("\n");

    Refresh(false);
    Update();
}

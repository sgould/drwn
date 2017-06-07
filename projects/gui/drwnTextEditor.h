/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextEditor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Generic text editor GUI.
**
*****************************************************************************/

#pragma once

// wxWidgets
#include "wx/wx.h"
#include "wx/utils.h"
#include "wx/dialog.h"
#include "wx/textctrl.h"

using namespace std;

// drwnTextEditor ------------------------------------------------------------

class drwnTextEditor : public wxDialog {
 protected:
    wxTextCtrl *_text;

 public:
    drwnTextEditor(wxWindow *parent, const char *title, bool bReadOnly = true);
    virtual ~drwnTextEditor();

    // i/o
    void clear();
    void addLine();
    void addLine(const char *str, bool bBold = false);
};

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOptionsEditor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Darwin GUI editor for derived drwnProperties classes.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"

#include "wx/wx.h"
#include "wx/utils.h"
#include "wx/dialog.h"
#include "wx/grid.h"

// drwnOptionsEditor ---------------------------------------------------------

class drwnOptionsEditor : public wxDialog
{
 protected:
    drwnProperties *_properties;
    drwnPropertiesCopy _copy;
    wxGrid *_grid;

 public:
    drwnOptionsEditor(wxWindow *parent, drwnProperties *properties);
    ~drwnOptionsEditor();

    // event callbacks
    void onSize(wxSizeEvent &event);
    void onBtnOkay(wxCommandEvent& event);
    void onDClick(wxGridEvent& event);

 protected:
    DECLARE_EVENT_TABLE()
};

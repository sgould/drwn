/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatrixEditor.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Darwin GUI editor for Eigen::VectorXd and Eigen::MatrixXd.
**
*****************************************************************************/

#pragma once

#include "wx/wx.h"
#include "wx/utils.h"
#include "wx/dialog.h"
#include "wx/grid.h"
#include "wx/spinctrl.h"
#include "wx/textctrl.h"

#include "Eigen/Core"

using namespace std;
using namespace Eigen;

enum {
    EDIT_TEXT_MODE = wxID_HIGHEST + 100,

    ROWS_SPIN_CTRL = wxID_HIGHEST + 200,
    COLS_SPIN_CTRL = wxID_HIGHEST + 210
};

// drwnMatrixEditor ----------------------------------------------------------

class drwnMatrixEditor : public wxDialog
{
 protected:
    bool _fixedRows;
    bool _fixedCols;
    wxGrid *_grid;
    wxTextCtrl *_text;
    
    // TODO: _readOnly
    
 public:
    drwnMatrixEditor(wxWindow *parent, const Eigen::MatrixXd *m, bool bFixed = true);
    drwnMatrixEditor(wxWindow *parent, const Eigen::VectorXd *v, bool bFixed = true);
    ~drwnMatrixEditor();

    // access
    int getRows() const { return _grid->GetNumberRows(); }
    int getCols() const { return _grid->GetNumberCols(); }

    Eigen::VectorXd getRow(int r = 0) const;
    Eigen::VectorXd getCol(int c = 0) const;
    Eigen::MatrixXd getMatrix() const;

    // event callbacks
    void onBtnOkay(wxCommandEvent& event);
    void onSpinCtrl(wxSpinEvent& event);
    void onChangeMode(wxCommandEvent& event);

 protected:
    DECLARE_EVENT_TABLE()
};

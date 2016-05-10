/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatrixEditor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "wx/wxprec.h"
#include "wx/utils.h"
#include "wx/sizer.h"
#include "wx/tglbtn.h"

#include "drwnBase.h"
#include "drwnMatrixEditor.h"

// Event Tables --------------------------------------------------------------

BEGIN_EVENT_TABLE(drwnMatrixEditor, wxDialog)
    EVT_BUTTON(wxID_OK, drwnMatrixEditor::onBtnOkay)
    EVT_TOGGLEBUTTON(EDIT_TEXT_MODE, drwnMatrixEditor::onChangeMode)
    EVT_SPINCTRL(ROWS_SPIN_CTRL, drwnMatrixEditor::onSpinCtrl)
    EVT_SPINCTRL(COLS_SPIN_CTRL, drwnMatrixEditor::onSpinCtrl)
END_EVENT_TABLE()

// drwnMatrixEditor ----------------------------------------------------------

drwnMatrixEditor::drwnMatrixEditor(wxWindow *parent, const Eigen::MatrixXd *m, bool bFixed) :
    wxDialog(parent, wxID_ANY, "Matrix Editor", wxDefaultPosition,wxDefaultSize,
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxFRAME_TOOL_WINDOW),
    _fixedRows(bFixed), _fixedCols(bFixed), _grid(NULL), _text(NULL)
{
    DRWN_ASSERT(m != NULL);
    wxBoxSizer *mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->SetMinSize(wxSize(120, 160));
    wxBoxSizer *buttons;

    // add rows and column button
    if (!bFixed) {
        buttons = new wxBoxSizer(wxHORIZONTAL);
        buttons->Add(new wxStaticText(this, wxID_ANY, "&rows"), 0, wxALL, 10);
        buttons->Add(new wxSpinCtrl(this, ROWS_SPIN_CTRL,
            toString(m->rows()), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 1024), 0, wxALL, 10);
        buttons->Add(new wxStaticText(this, wxID_ANY, "&cols"), 0, wxALL, 10);
        buttons->Add(new wxSpinCtrl(this, COLS_SPIN_CTRL,
            toString(m->cols()), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 1024), 0, wxALL, 10);
        buttons->Add(new wxToggleButton(this, EDIT_TEXT_MODE, "&Text Mode"), 0, wxALL, 10);
        mainSizer->Add(buttons, 0, wxALIGN_RIGHT);
    }

    // add grid input
    _grid = new wxGrid(this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
        wxEXPAND | wxWANTS_CHARS);

    _grid->CreateGrid(m->rows(), m->cols());
    for (int i = 0; i < m->rows(); i++) {
        for (int j = 0; j < m->cols(); j++) {
            _grid->SetCellValue(i, j, toString((*m)(i, j)));

            // color cell
            _grid->SetCellBackgroundColour(i, j, 
                wxColour(0xff, (i % 2 == 0) ? 0xff : 0xe0, (j % 2 == 0) ? 0xff : 0xe0));
        }
    }

    mainSizer->Add(_grid, 1, wxEXPAND | wxALL, 10);

    // add text input
    _text = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
        wxEXPAND | wxWANTS_CHARS | wxTE_DONTWRAP | wxTE_MULTILINE | wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB);
    _text->Show(false);
    mainSizer->Add(_text, 1, wxEXPAND | wxALL, 10);

    // TODO: change to CreateButtonSizer?
    buttons = new wxBoxSizer(wxHORIZONTAL);
#if 0
    // TODO: wxWidgets version problem?
    buttons->Add(new wxButton(this, wxID_OK, "OK"),
        wxSizerFlags(0).Align().Border(wxALL, 10));
    buttons->Add(new wxButton( this, wxID_CANCEL, "Cancel" ),
        wxSizerFlags(0).Align().Border(wxALL, 10));
#else
    buttons->Add(new wxButton(this, wxID_OK, "OK"), 0, wxALL, 10);
    buttons->Add(new wxButton(this, wxID_CANCEL, "Cancel"), 0, wxALL, 10);
#endif

    mainSizer->Add(buttons, 0, wxALIGN_CENTER);

    // implement the sizer
    SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);
}

drwnMatrixEditor::drwnMatrixEditor(wxWindow *parent, const Eigen::VectorXd *v, bool bFixed) :
    wxDialog(parent, wxID_ANY, "Vector Editor", wxDefaultPosition,wxDefaultSize,
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxFRAME_TOOL_WINDOW),
    _fixedRows(bFixed), _fixedCols(true), _grid(NULL), _text(NULL)
{
    DRWN_ASSERT(v != NULL);
    wxBoxSizer *mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->SetMinSize(wxSize(120, 160));
    wxBoxSizer *buttons;

    // add vector rows button
    if (!bFixed) {
        buttons = new wxBoxSizer(wxHORIZONTAL);
        buttons->Add(new wxStaticText(this, wxID_ANY, "&rows"), 0, wxALL, 10);
        buttons->Add(new wxSpinCtrl(this, ROWS_SPIN_CTRL,
            toString(v->rows()), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 1024), 0, wxALL, 10);
        buttons->Add(new wxToggleButton(this, EDIT_TEXT_MODE, "&Text Mode"), 0, wxALL, 10);
        mainSizer->Add(buttons, 0, wxALIGN_RIGHT);
    }

    // add grid input
    _grid = new wxGrid(this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
        wxEXPAND | wxWANTS_CHARS);

    _grid->CreateGrid(v->rows(), 1);
    for (int i = 0; i < v->rows(); i++) {
        _grid->SetCellValue(i, 0, toString((*v)[i]));
    }
    _grid->AutoSizeRows();

    mainSizer->Add(_grid, 1, wxEXPAND | wxALL, 10);

    // add text input
    // TODO

    // TODO: change to CreateButtonSizer?
    buttons = new wxBoxSizer(wxHORIZONTAL);
#if 0
    // TODO: wxWidgets version problem?
    buttons->Add(new wxButton(this, wxID_OK, "OK"),
        wxSizerFlags(0).Align().Border(wxALL, 10));
    buttons->Add(new wxButton( this, wxID_CANCEL, "Cancel" ),
        wxSizerFlags(0).Align().Border(wxALL, 10));
#else
    buttons->Add(new wxButton(this, wxID_OK, "OK"), 0, wxALL, 10);
    buttons->Add(new wxButton(this, wxID_CANCEL, "Cancel"), 0, wxALL, 10);
#endif

    mainSizer->Add(buttons, 0, wxALIGN_CENTER);

    // implement the sizer
    SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);
}

drwnMatrixEditor::~drwnMatrixEditor()
{
    // do nothing
}

// access
Eigen::VectorXd drwnMatrixEditor::getRow(int r) const
{
    DRWN_ASSERT((r >= 0) && (r < _grid->GetNumberRows()));

    Eigen::VectorXd v = Eigen::VectorXd::Zero(_grid->GetNumberCols());
    for (int c = 0; c < _grid->GetNumberCols(); c++) {
        v[c] = atof(_grid->GetCellValue(r, c).c_str());
    }

    return v;
}

Eigen::VectorXd drwnMatrixEditor::getCol(int c) const
{
    DRWN_ASSERT((c >= 0) && (c < _grid->GetNumberCols()));

    Eigen::VectorXd v = Eigen::VectorXd::Zero(_grid->GetNumberRows());
    for (int r = 0; r < _grid->GetNumberRows(); r++) {
        v[r] = atof(_grid->GetCellValue(r, c).c_str());
    }

    return v;
}

Eigen::MatrixXd drwnMatrixEditor::getMatrix() const
{
    Eigen::MatrixXd m(_grid->GetNumberRows(), _grid->GetNumberCols());
    for (int r = 0; r < _grid->GetNumberRows(); r++) {
        for (int c = 0; c < _grid->GetNumberCols(); c++) {
            m(r, c) = atof(_grid->GetCellValue(r, c).c_str());
        }
    }

    return m;
}

// event callbacks
void drwnMatrixEditor::onBtnOkay(wxCommandEvent& event)
{
    // TODO?
    // no nothing
    event.Skip();
    return;
}

void drwnMatrixEditor::onSpinCtrl(wxSpinEvent& event)
{
    if (event.GetId() == ROWS_SPIN_CTRL) {
        int newRows = event.GetPosition() - _grid->GetNumberRows(); 
        if (newRows > 0) {
            _grid->AppendRows(newRows);
            for (int i = 0; i < newRows; i++) {
                for (int j = 0; j < _grid->GetNumberCols(); j++) {
                    _grid->SetCellValue(_grid->GetNumberRows() - i - 1, j, "0.0");
                }
            }
        } else if (newRows < 0) {
            _grid->DeleteRows(0, -newRows);
        }
    } else if (event.GetId() == COLS_SPIN_CTRL) {
        int newCols = event.GetPosition() - _grid->GetNumberCols(); 
        if (newCols > 0) {
            _grid->AppendCols(newCols);
            for (int i = 0; i < _grid->GetNumberRows(); i++) {
                for (int j = 0; j < newCols; j++) {
                    _grid->SetCellValue(i, _grid->GetNumberCols() - j - 1, "0.0");
                }
            }
        } else if (newCols < 0) {
            _grid->DeleteCols(0, -newCols);
        }
    }
}

void drwnMatrixEditor::onChangeMode(wxCommandEvent& event)
{
    if (event.GetId() == EDIT_TEXT_MODE) {
        _text->Show(event.IsChecked());
        _grid->Show(!event.IsChecked());
    }

    // update text
    if (_text->IsShown()) {
        MatrixXd m = this->getMatrix();
        _text->Clear();
        std::stringstream s;
        s << m << endl;
        _text->AppendText(wxString(s.str()));
    }

    this->GetSizer()->Layout();
    event.Skip();
    return;
}

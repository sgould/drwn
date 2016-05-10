/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnOptionsEditor.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "wx/wxprec.h"
#include "wx/utils.h"
#include "wx/sizer.h"

#include "drwnBase.h"

#include "drwnOptionsEditor.h"
#include "drwnMatrixEditor.h"

// Event Tables --------------------------------------------------------------

BEGIN_EVENT_TABLE(drwnOptionsEditor, wxDialog)
    EVT_SIZE(drwnOptionsEditor::onSize)
    EVT_BUTTON(wxID_OK, drwnOptionsEditor::onBtnOkay)
    EVT_GRID_CELL_LEFT_DCLICK(drwnOptionsEditor::onDClick)
END_EVENT_TABLE()

// drwnOptionsEditor ---------------------------------------------------------

// Comment: instantiating wxDialog(parent, ...) instead of wxDialog(NULL, ...)
// causes change of focus events in the dialog editor (and children, e.g. matrix
// editor) to generate scroll events in the parent.

drwnOptionsEditor::drwnOptionsEditor(wxWindow *parent, drwnProperties *properties) :
    wxDialog(NULL, wxID_ANY, "Properties", wxDefaultPosition,wxDefaultSize,
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxFRAME_TOOL_WINDOW),
    _properties(properties), _copy(properties), _grid(NULL)
{
    wxBoxSizer *mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->SetMinSize(wxSize(120, 160));

    // add properties grid
    _grid = new wxGrid(this, wxID_ANY, wxDefaultPosition, wxDefaultSize,
        wxEXPAND | wxWANTS_CHARS);
    _grid->CreateGrid(_properties->numProperties(), 1);
    _grid->SetColLabelValue(0, "Value");
    for (unsigned i = 0; i < _properties->numProperties(); i++) {
        _grid->SetRowLabelValue(i, _properties->getPropertyName(i).c_str());
        _grid->SetCellValue(i, 0, _properties->getPropertyAsString(i).c_str());
        _grid->SetCellBackgroundColour(i, 0, *wxWHITE);
        if (_properties->isReadOnly(i)) {
            _grid->SetReadOnly(i, 0);
            _grid->SetCellTextColour(i, 0, *wxRED);
        }
        
        switch (_properties->getPropertyType(i)) {
        case DRWN_BOOLEAN_PROPERTY:
            // TODO
            break;
        case DRWN_SELECTION_PROPERTY:
            {
                const vector<string> *choices = 
                    ((drwnSelectionProperty *)_properties->getProperty(i))->getChoices();
                wxArrayString choiceArray;
                for (unsigned k = 0; k < choices->size(); k++) {
                    choiceArray.Add(wxString(choices->at(k)));
                }
                _grid->SetCellEditor(i, 0, new wxGridCellChoiceEditor(choiceArray));
            }
            break;
        case DRWN_VECTOR_PROPERTY:
            _grid->SetReadOnly(i, 0);
            break;
        case DRWN_MATRIX_PROPERTY:
            _grid->SetReadOnly(i, 0);
            break;
        default:
            // do nothing: happy with text editor
            break;
        }
    }
    
    //_grid->AutoSizeColumns();
    _grid->AutoSize();
    mainSizer->Add(_grid, 1, wxEXPAND | wxALL, 10);

    // TODO: change to CreateButtonSizer?
    wxBoxSizer *buttons = new wxBoxSizer(wxHORIZONTAL);
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

drwnOptionsEditor::~drwnOptionsEditor()
{
    // do nothing
}

// event callbacks
void drwnOptionsEditor::onSize(wxSizeEvent& event)
{
    int width, height;
    GetClientSize(&width, &height);
    // TODO: resize grid
    event.Skip();
}

void drwnOptionsEditor::onBtnOkay(wxCommandEvent& event)
{
    DRWN_ASSERT(_properties != NULL);
    _copy.copyBack(_properties);

    // TODO: implement in the same way as vectors and matrices
    for (unsigned i = 0; i < _properties->numProperties(); i++) {
        if (_properties->isReadOnly(i)) continue;
        switch (_properties->getPropertyType(i)) {
        case DRWN_BOOLEAN_PROPERTY:
        case DRWN_INTEGER_PROPERTY:
        case DRWN_DOUBLE_PROPERTY:
        case DRWN_STRING_PROPERTY:
        case DRWN_FILENAME_PROPERTY:
        case DRWN_DIRECTORY_PROPERTY:
        case DRWN_SELECTION_PROPERTY:
            _properties->setProperty(i, string(_grid->GetCellValue(i, 0)));
            break;
        default:
            // not implemented yet
            break;
        }
    }
    
    event.Skip();
}

void drwnOptionsEditor::onDClick(wxGridEvent& event)
{
    unsigned indx = (unsigned)event.GetRow();
    DRWN_ASSERT(indx < _copy.numProperties());

    bool bReadOnly = _copy.isReadOnly(indx);
    if (_copy.getPropertyType(indx) == DRWN_MATRIX_PROPERTY) {
        drwnMatrixEditor dlg(this, &_copy.getMatrixProperty(indx), bReadOnly);
        if ((dlg.ShowModal() == wxID_OK) && !bReadOnly) {
            Eigen::MatrixXd m = dlg.getMatrix();
            _copy.setProperty(indx, m);
            _grid->SetCellValue(indx, 0, _copy.getPropertyAsString(indx));
        }
    } else if (_copy.getPropertyType(indx) == DRWN_VECTOR_PROPERTY) {
        drwnMatrixEditor dlg(this, &_copy.getVectorProperty(indx), bReadOnly);
        if ((dlg.ShowModal() == wxID_OK) && !bReadOnly) {
            Eigen::VectorXd v = dlg.getCol();
            _copy.setProperty(indx, v);
            _grid->SetCellValue(indx, 0, _copy.getPropertyAsString(indx));
        }
    } else if (_copy.getPropertyType(indx) == DRWN_FILENAME_PROPERTY) {
        wxFileDialog dlg(this, string("Choose a file for ") + _copy.getPropertyName(indx),
            "", _grid->GetCellValue(indx, 0), "*.*", wxFD_OPEN);
        if (dlg.ShowModal() == wxID_OK) {
            _grid->SetCellValue(indx, 0, dlg.GetPath());
        }
    } else if (_copy.getPropertyType(indx) == DRWN_DIRECTORY_PROPERTY) {
        wxDirDialog dlg(this, string("Choose a directory for ") + _copy.getPropertyName(indx),
            _grid->GetCellValue(indx, 0), wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);
        if (dlg.ShowModal() == wxID_OK) {
            _grid->SetCellValue(indx, 0, dlg.GetPath());
        }
    }

    event.Skip();    
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImportFilesNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implements nodes for importing or exporting data in multiple files with
**  filename of the form <key>.<ext> all within the same directory. Data can
**  be formatted as (8, 32 or 64-bit) binary or text.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnImportExportFilesBase -------------------------------------------------

class drwnImportExportFilesBase : public drwnNode {
 protected:
    static vector<string> _formats;

    string _directory;      // source directory
    string _extension;      // source extension
    int _fileFormat;        // data format (double, float, int) 

 public:
    drwnImportExportFilesBase(const char *name = NULL, drwnGraph *owner = NULL);
    drwnImportExportFilesBase(const drwnImportExportFilesBase& node);
    virtual ~drwnImportExportFilesBase();
};

// drwnImportFilesNode -------------------------------------------------------

class drwnImportFilesNode : public drwnImportExportFilesBase {
 protected:
    int _nFeatures;         // number of features
    int _nHeaderBytes;      // number of header bytes
    int _nFooterBytes;      // number of footer bytes

 public:
    drwnImportFilesNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnImportFilesNode(const drwnImportFilesNode& node);
    virtual ~drwnImportFilesNode();

    // i/o
    const char *type() const { return "drwnImportFilesNode"; }
    drwnImportFilesNode *clone() const { return new drwnImportFilesNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();

 protected:
    void importTextFile(const string& key);
    void importBinaryFile(const string& key);
};

// drwnExportFilesNode -------------------------------------------------------

class drwnExportFilesNode : public drwnImportExportFilesBase {
 public:
    drwnExportFilesNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnExportFilesNode(const drwnExportFilesNode& node);
    virtual ~drwnExportFilesNode();

    // i/o
    const char *type() const { return "drwnExportFilesNode"; }
    drwnExportFilesNode *clone() const { return new drwnExportFilesNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();

 protected:
    void exportTextFile(const string& filename, const drwnDataRecord *record) const;
    void exportBinaryFile(const string& filename, const drwnDataRecord *record) const;
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnImportFilesNode);
DRWN_DECLARE_AUTOREGISTERNODE(drwnExportFilesNode);


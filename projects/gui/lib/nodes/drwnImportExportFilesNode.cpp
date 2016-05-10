/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnImportFilesNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnImportExportFilesNode.h"

using namespace std;
using namespace Eigen;

// drwnImportExportFilesBase -------------------------------------------------

vector<string> drwnImportExportFilesBase::_formats;

drwnImportExportFilesBase::drwnImportExportFilesBase(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _directory(""), _extension(".bin"), _fileFormat(1)
{
    // define formats if not already done
    if (_formats.empty()) {
        _formats.push_back(string("text"));
        _formats.push_back(string("double (64-bit)"));
        _formats.push_back(string("float (32-bit)"));
        _formats.push_back(string("int (32-bit)"));
    }

    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
    declareProperty("format", new drwnSelectionProperty(&_fileFormat, &_formats));
}

drwnImportExportFilesBase::drwnImportExportFilesBase(const drwnImportExportFilesBase& node) :
    drwnNode(node), _directory(node._directory), _extension(node._extension),
    _fileFormat(node._fileFormat)
{
    // declare propertys
    declareProperty("directory", new drwnDirectoryProperty(&_directory));
    declareProperty("extension", new drwnStringProperty(&_extension));
    declareProperty("format", new drwnSelectionProperty(&_fileFormat, &_formats));
}

drwnImportExportFilesBase::~drwnImportExportFilesBase()
{
    // do nothing
}

// drwnImportFilesNode -------------------------------------------------------

drwnImportFilesNode::drwnImportFilesNode(const char *name, drwnGraph *owner) :
    drwnImportExportFilesBase(name, owner), _nFeatures(1), _nHeaderBytes(0),
    _nFooterBytes(0)
{
    _nVersion = 100;
    _desc = "Imports data from multiple binary or text files";

    // define ports
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D matrix of data"));

    // declare propertys
    declareProperty("features", new drwnIntegerProperty(&_nFeatures));
    declareProperty("headerBytes", new drwnIntegerProperty(&_nHeaderBytes));
    declareProperty("footerBytes", new drwnIntegerProperty(&_nFooterBytes));
}

drwnImportFilesNode::drwnImportFilesNode(const drwnImportFilesNode& node) :
    drwnImportExportFilesBase(node), _nFeatures(node._nFeatures),
    _nHeaderBytes(node._nHeaderBytes), _nFooterBytes(node._nFooterBytes)
{
    // declare propertys
    declareProperty("features", new drwnIntegerProperty(&_nFeatures));
    declareProperty("headerBytes", new drwnIntegerProperty(&_nHeaderBytes));
    declareProperty("footerBytes", new drwnIntegerProperty(&_nFooterBytes));
}

drwnImportFilesNode::~drwnImportFilesNode()
{
    // do nothing
}

// processing
void drwnImportFilesNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnImportFilesNode::updateForwards()
{
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    // list all matching files in the directory
    vector<string> keys = drwnDirectoryListing(_directory.c_str(),
        _extension.c_str(), false, false);
    if (keys.empty()) {
        DRWN_LOG_WARNING("no files found in \"" << _directory << "\" for node \"" << _name << "\"");
        return;
    }

    // import data
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        if (tblOut->hasKey(*it)) continue;

        // load data into record
        switch (_fileFormat) {
        case 0: // text
            importTextFile(*it);
            break;
        case 1: // double
        case 2: // float
        case 3: // integer
            importBinaryFile(*it);
            break;
        default:
            DRWN_LOG_FATAL("unknown file format " << _fileFormat
                << " in node \"" << _name << "\"");
        }
    }
    DRWN_END_PROGRESS;
}

void drwnImportFilesNode::importTextFile(const string& key)
{
    string filename = _directory + DRWN_DIRSEP + key + _extension;

    ifstream ifs(filename.c_str());
    DRWN_ASSERT(!ifs.fail());

    vector<vector<double> > data;
    while (1) {
        vector<double> dataRow(_nFeatures);
        for (int i = 0; i < _nFeatures; i++) {
            ifs >> dataRow[i];
        }
        if (ifs.fail()) break;

        data.push_back(dataRow);
    }
    ifs.close();

    // create record
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    drwnDataRecord *recordOut = tblOut->lockRecord(key);
    recordOut->data() = MatrixXd::Zero(data.size(), _nFeatures);

    for (int i = 0; i < (int)data.size(); i++) {
        recordOut->data().row(i) = Eigen::Map<MatrixXd>(&data[i][0], 1, _nFeatures);
    }

    // unlock record
    tblOut->unlockRecord(key);
}

void drwnImportFilesNode::importBinaryFile(const string& key)
{
    string filename = _directory + DRWN_DIRSEP + key + _extension;

    ifstream ifs(filename.c_str(), ios::binary);
    DRWN_ASSERT(!ifs.fail());
    ifs.seekg(0, ios::end);
    int numFeatureBytes = (int)ifs.tellg() - _nHeaderBytes - _nFooterBytes;
    int nObs = 0;
    switch (_fileFormat) {
    case 1: // double
        nObs = numFeatureBytes / (int)(_nFeatures * sizeof(double));
        break;
    case 2: // float
        nObs = numFeatureBytes / (int)(_nFeatures * sizeof(float));
        break;
    case 3: // int
        nObs = numFeatureBytes / (int)(_nFeatures * sizeof(int));
        break;
    default:
        DRWN_LOG_FATAL("unknown binary data type");
    }

    DRWN_LOG_DEBUG(numFeatureBytes << " bytes of features in " << filename
        << " (" << nObs << " observations)");
    if (nObs < 1) {
        DRWN_LOG_WARNING("file for record " << key << " has no data");
        ifs.close();
        return;
    }

    if (numFeatureBytes % nObs != 0) {
        DRWN_LOG_WARNING("extra bytes in file \"" << filename << "\"");
    }

    // create record
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    drwnDataRecord *recordOut = tblOut->lockRecord(key);
    recordOut->data() = MatrixXd::Zero(nObs, _nFeatures);

    // import data
    ifs.seekg(_nHeaderBytes, ios::beg);
    switch (_fileFormat) {
    case 1: // double
        {
            vector<double> d(recordOut->numFeatures());
            for (int i = 0; i < recordOut->numObservations(); i++) {
                ifs.read((char *)&d[0], d.size() * sizeof(double));
                recordOut->data().row(i) = Eigen::Map<MatrixXd>(&d[0], 1, d.size());
            }
        }
        break;
    case 2: // float
        {
            vector<float> f(recordOut->numFeatures());
            for (int i = 0; i < recordOut->numObservations(); i++) {
                ifs.read((char *)&f[0], f.size() * sizeof(float));
                recordOut->data().row(i) = Eigen::Map<MatrixXf>(&f[0], 1, f.size()).cast<double>();
            }
        }
        break;
    case 3: // int
        {
            vector<int> n(recordOut->numFeatures());
            for (int i = 0; i < recordOut->numObservations(); i++) {
                ifs.read((char *)&n[0], n.size() * sizeof(int));
                recordOut->data().row(i) = Eigen::Map<MatrixXi>(&n[0], 1, n.size()).cast<double>();
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unknown binary data type");
    }

    ifs.close();

    // unlock record
    tblOut->unlockRecord(key);
}

// drwnExportFilesNode -------------------------------------------------------

drwnExportFilesNode::drwnExportFilesNode(const char *name, drwnGraph *owner) :
    drwnImportExportFilesBase(name, owner)
{
    _nVersion = 100;
    _desc = "Exports individual records to binary or text files";

    // define ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D matrix of data"));
}

drwnExportFilesNode::drwnExportFilesNode(const drwnExportFilesNode& node) :
    drwnImportExportFilesBase(node)
{
    // do nothing
}

drwnExportFilesNode::~drwnExportFilesNode()
{
    // do nothing
}

// processing
void drwnExportFilesNode::evaluateForwards()
{
    drwnDataTable *tbl = _inputPorts[0]->getTable();
    if (tbl == NULL) {
        DRWN_LOG_WARNING("node \"" << _name << "\" has no input");
        return;
    }

    // create output directory
    if (!drwnDirExists(_directory.c_str())) {
        drwnCreateDirectory(_directory.c_str());
    }

    // interate over records
    vector<string> keys = tbl->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        string filename = _directory + DRWN_DIRSEP + *it + _extension;
        const drwnDataRecord *record = tbl->lockRecord(*it);
        DRWN_ASSERT(record != NULL);

        // write data
        switch (_fileFormat) {
        case 0: // text
            exportTextFile(filename, record);
            break;
        case 1: // double
        case 2: // float
        case 3: // integer
            exportBinaryFile(filename, record);
            break;
        default:
            DRWN_LOG_FATAL("unknown file format in node \"" << _name << "\"");
        }

        tbl->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnExportFilesNode::updateForwards()
{
    drwnDataTable *tbl = _inputPorts[0]->getTable();
    if (tbl == NULL) {
        DRWN_LOG_WARNING("node \"" << _name << "\" has no input");
        return;
    }

    // create output directory
    if (!drwnDirExists(_directory.c_str())) {
        drwnCreateDirectory(_directory.c_str());
    }

    // interate over records
    vector<string> keys = tbl->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        string filename = _directory + DRWN_DIRSEP + *it + _extension;
        if (drwnFileExists(filename.c_str())) {
            DRWN_LOG_VERBOSE("skipping existing file \"" << filename << "\"");
            continue;
        }

        const drwnDataRecord *record = tbl->lockRecord(*it);
        DRWN_ASSERT(record != NULL);

        // write data
        switch (_fileFormat) {
        case 0: // text
            exportTextFile(filename, record);
            break;
        case 1: // double
        case 2: // float
        case 3: // integer
            exportBinaryFile(filename, record);
            break;
        default:
            DRWN_LOG_FATAL("unknown file format " << _fileFormat
                << " in node \"" << _name << "\"");
        }

        tbl->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}

void drwnExportFilesNode::exportTextFile(const string& filename,
    const drwnDataRecord *record) const
{
    ofstream ofs(filename.c_str());
    ofs << record->data() << endl;
    if (ofs.fail()) {
        DRWN_LOG_ERROR("an error occurred writing to text file \"" << filename << "\"");
    }
    ofs.close();
}

void drwnExportFilesNode::exportBinaryFile(const string& filename,
    const drwnDataRecord *record) const
{
    ofstream ofs(filename.c_str(), ios::binary);

    switch (_fileFormat) {
    case 1: // double
        {
            vector<double> d(record->numFeatures());
            for (int i = 0; i < record->numObservations(); i++) {
                Eigen::Map<MatrixXd>(&d[0], 1, d.size()) = record->data().row(i);
                ofs.write((char *)&d[0], d.size() * sizeof(double));
            }
        }
        break;
    case 2: // float
        {
            vector<float> f(record->numFeatures());
            for (int i = 0; i < record->numObservations(); i++) {
                Eigen::Map<MatrixXf>(&f[0], 1, f.size()) = record->data().row(i).cast<float>();
                ofs.write((char *)&f[0], f.size() * sizeof(float));
            }
        }
        break;
    case 3: // int
        {
            vector<int> n(record->numFeatures());
            for (int i = 0; i < record->numObservations(); i++) {
                Eigen::Map<MatrixXi>(&n[0], 1, n.size()) = record->data().row(i).cast<int>();
                ofs.write((char *)&n[0], n.size() * sizeof(int));
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unknown binary data type");
    }

    if (ofs.fail()) {
        DRWN_LOG_ERROR("an error occurred writing to binary file \"" << filename << "\"");
    }
    ofs.close();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Source", drwnImportFilesNode);
DRWN_AUTOREGISTERNODE("Sink", drwnExportFilesNode);

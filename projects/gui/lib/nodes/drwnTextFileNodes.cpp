/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTextFileNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnTextFileNodes.h"

using namespace std;
using namespace Eigen;

// drwnTextFileSourceNode ----------------------------------------------------

drwnTextFileSourceNode::drwnTextFileSourceNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _filename(""), _delimiter(" "), _numHeaderLines(0),
    _bIncludesKey(true), _keyPrefix("")
{
    _nVersion = 100;
    _desc = "Reads data from a text file";

    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D matrix of data"));

    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("delimiter", new drwnStringProperty(&_delimiter));
    declareProperty("headerLines", new drwnIntegerProperty(&_numHeaderLines));
    declareProperty("includesKey", new drwnBooleanProperty(&_bIncludesKey));
    declareProperty("keyPrefix", new drwnStringProperty(&_keyPrefix));
}

drwnTextFileSourceNode::drwnTextFileSourceNode(const drwnTextFileSourceNode& node) :
    drwnNode(node), _filename(node._filename), _delimiter(node._delimiter),
    _numHeaderLines(node._numHeaderLines), _bIncludesKey(node._bIncludesKey),
    _keyPrefix(node._keyPrefix)
{
    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("delimiter", new drwnStringProperty(&_delimiter));
    declareProperty("headerLines", new drwnIntegerProperty(&_numHeaderLines));
    declareProperty("includesKey", new drwnBooleanProperty(&_bIncludesKey));
    declareProperty("keyPrefix", new drwnStringProperty(&_keyPrefix));
}

drwnTextFileSourceNode::~drwnTextFileSourceNode()
{
    // do nothing
}

// processing
void drwnTextFileSourceNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnTextFileSourceNode::updateForwards()
{
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    if (_filename.empty()) {
        DRWN_LOG_WARNING("no source filename in node \"" << _name << "\"");
        return;
    }

    // build set of pre-existing keys
    // TODO: avoid the need for the vector
    vector<string> tableKeys = tblOut->getKeys();
    set<string> existingRecords(tableKeys.begin(), tableKeys.end());

    // open file
    ifstream ifs(_filename.c_str());
    if (ifs.fail()) {
        DRWN_LOG_ERROR("could not open file \"" << _filename << "\" in node \"" << _name << "\"");
        return;
    }

    // get file size (for progress)
    ifs.seekg(0, ios::end);
    double fileSize = (double)ifs.tellg();
    ifs.seekg(0, ios::beg);

    // skip header lines
    for (int i = 0; i < _numHeaderLines; i++) {
        ifs.ignore(DRWN_INT_MAX, '\n');
        if (ifs.eof()) break;
    }

    // read file
    int lineNum = _numHeaderLines;
    while (!ifs.eof()) {
        DRWN_SET_PROGRESS((double)ifs.tellg() / fileSize);

        // read next line
        lineNum += 1;
        string nextLine;
        getline(ifs, nextLine);
        drwn::trim(nextLine);
        if (nextLine.empty()) {
            continue;
        }

        // extract key
        string key;
        if (_bIncludesKey) {
            size_t searchPos = nextLine.find(_delimiter, 0);
            if (searchPos == string::npos) {
                DRWN_LOG_ERROR("could not parse line " << lineNum
                    << " in file \"" << _filename << "\"");
                break;
            }

            key = _keyPrefix + nextLine.substr(0, searchPos);
            drwn::trim(key);
            if (existingRecords.find(key) != existingRecords.end()) {
                continue;
            }

            nextLine = nextLine.substr(searchPos + _delimiter.length(), string::npos);
        } else {
            //key = strBaseName(_filename) + drwn::padString(toString(lineNum), 8);
            key = _keyPrefix + drwn::padString(toString(lineNum), 8);
        }

        // parse features
        vector<double> v;
        drwn::parseString<double>(_delimiter.compare(" ") == 0 ? nextLine :
            drwn::strReplaceSubstr(nextLine, _delimiter, string(" ")), v);

        // add to record
        drwnDataRecord *recordOut = tblOut->lockRecord(key);
        DRWN_ASSERT(recordOut != NULL);
        if ((recordOut->data().rows() != 0) && (recordOut->data().cols() != (int)v.size())) {
            DRWN_LOG_ERROR("data size mismatch on line " << lineNum
                << " of file \"" << _filename << "\"");
            tblOut->unlockRecord(key);
            break;
        }

        if (recordOut->isEmpty()) {
            recordOut->data() = Eigen::Map<MatrixXd>(&v[0], 1, v.size());
        } else {
            MatrixXd newData(recordOut->data().rows() + 1, v.size());
            newData.block(0, 0, recordOut->data().rows(), v.size()) = recordOut->data();
            newData.block(recordOut->data().rows(), 0, 1, v.size()) = Eigen::Map<VectorXd>(&v[0], v.size()).transpose();
            recordOut->data() = newData;
        }
        tblOut->unlockRecord(key);
    }

    ifs.close();
    DRWN_END_PROGRESS;
}

// drwnTextFileSinkNode ------------------------------------------------------

drwnTextFileSinkNode::drwnTextFileSinkNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _filename(""), _delimiter(" "), _bIncludeHeader(true),
    _bIncludeKey(true)
{
    _nVersion = 100;
    _desc = "Writes data to a text file";

    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D matrix of data"));

    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("delimiter", new drwnStringProperty(&_delimiter));
    declareProperty("includeHeader", new drwnBooleanProperty(&_bIncludeHeader));
    declareProperty("includeKey", new drwnBooleanProperty(&_bIncludeKey));
}

drwnTextFileSinkNode::drwnTextFileSinkNode(const drwnTextFileSinkNode& node) :
    drwnNode(node), _filename(node._filename), _delimiter(node._delimiter),
    _bIncludeHeader(node._bIncludeHeader), _bIncludeKey(node._bIncludeKey)
{
    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("delimiter", new drwnStringProperty(&_delimiter));
    declareProperty("includeHeader", new drwnBooleanProperty(&_bIncludeHeader));
    declareProperty("includeKey", new drwnBooleanProperty(&_bIncludeKey));
}

drwnTextFileSinkNode::~drwnTextFileSinkNode()
{
    // do nothing
}

// processing
void drwnTextFileSinkNode::evaluateForwards()
{
    ofstream ofs(_filename.c_str());
    if (ofs.fail()) {
        DRWN_LOG_ERROR("node \"" << _name << "\" could not open file " << _filename);
        return;
    }

    if (_bIncludeHeader) {
        // TODO: write table fieldnames
        ofs << "KEY" << _delimiter << "DATA\n";
    }

    drwnDataTable *tbl = _inputPorts[0]->getTable();
    if (tbl == NULL) {
        DRWN_LOG_WARNING("node \"" << _name << "\" has no input");
        ofs.close();
        return;
    }

    // interate over records
    vector<string> keys = tbl->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        drwnDataRecord *record = tbl->lockRecord(*it);
        DRWN_ASSERT(record != NULL);
        for (int i = 0; i < record->numObservations(); i++) {
            if (_bIncludeKey) {
                ofs << *it << _delimiter;
            }
            for (int j = 0; j < record->numFeatures(); j++) {
                if (j > 0) ofs << _delimiter;
                ofs << record->data()(i, j);
            }
            ofs << "\n";
        }
        tbl->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
    ofs.close();
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Source", drwnTextFileSourceNode);
DRWN_AUTOREGISTERNODE("Sink", drwnTextFileSinkNode);

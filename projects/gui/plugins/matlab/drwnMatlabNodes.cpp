/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabNode.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ Standard Libraries
#include <cstdlib>

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnML.h"
#include "drwnEngine.h"

// Matlab library
#include "engine.h"
#include "mat.h"

#include "drwnMatlabUtils.h"
#include "drwnMatlabNodes.h"
#include "drwnMatlabShell.h"

using namespace std;
using namespace Eigen;

// drwnMatlabNode statics and constants -----------------------------------

#ifdef __LINUX__
string drwnMatlabNode::STARTCMD("matlab -nosplash -nodesktop");
#else
string drwnMatlabNode::STARTCMD("");
#endif

Engine *drwnMatlabNode::_matlabEngine = NULL;
int drwnMatlabNode::_refCount = 0;

// drwnMatlabNode ---------------------------------------------------------

drwnMatlabNode::drwnMatlabNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _nInputs(1), _nOutputs(1), _colour(-1)
{
    _nVersion = 100;
    _desc = "Interface to Matlab";

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D data matrix"));

    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("numOutputs", new drwnIntegerProperty(&_nOutputs));
    declareProperty("fwdFunction", new drwnFilenameProperty(&_fwdFcnName));
    declareProperty("bckFunction", new drwnFilenameProperty(&_bckFcnName));
    declareProperty("initFunction", new drwnFilenameProperty(&_initFcnName));
    declareProperty("estFunction", new drwnFilenameProperty(&_estFcnName));
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("theta", new drwnMatrixProperty(&_theta));

    // increment reference count
    _refCount += 1;
}

drwnMatlabNode::drwnMatlabNode(const drwnMatlabNode& node) :
    drwnNode(node), _nInputs(node._nInputs), _nOutputs(node._nOutputs),
    _fwdFcnName(node._fwdFcnName), _bckFcnName(node._bckFcnName),
    _initFcnName(node._initFcnName), _estFcnName(node._estFcnName),
    _colour(node._colour), _theta(node._theta)
{
    // declare propertys
    declareProperty("numInputs", new drwnIntegerProperty(&_nInputs));
    declareProperty("numOutputs", new drwnIntegerProperty(&_nOutputs));
    declareProperty("fwdFunction", new drwnFilenameProperty(&_fwdFcnName));
    declareProperty("bckFunction", new drwnFilenameProperty(&_bckFcnName));
    declareProperty("initFunction", new drwnFilenameProperty(&_initFcnName));
    declareProperty("estFunction", new drwnFilenameProperty(&_estFcnName));
    declareProperty("colour", new drwnIntegerProperty(&_colour));
    declareProperty("theta", new drwnMatrixProperty(&_theta));

    // increment reference count
    _refCount += 1;
}

drwnMatlabNode::~drwnMatlabNode()
{
    // decrement reference count
    _refCount -= 1;
    DRWN_ASSERT(_refCount >= 0);

    // close connection to Matlab
    if ((_refCount == 0) && (_matlabEngine != NULL)) {
        engClose(_matlabEngine);
        _matlabEngine = NULL;
    }
}

// i/o
bool drwnMatlabNode::load(drwnXMLNode& xml)
{
    drwnNode::load(xml);
    updatePorts();
    return true;
}

// gui
void drwnMatlabNode::showWindow()
{
    if (_matlabEngine == NULL) {
        if (!(_matlabEngine = engOpen(STARTCMD.c_str()))) {
            DRWN_LOG_ERROR("can't start the Matlab engine with command \"" << STARTCMD << "\"");
            return;
        }
    }

    engSetVisible(_matlabEngine, 1);

    if (_window != NULL) {
        _window->Show();
    } else {
        _window = new drwnMatlabShell(this, _matlabEngine);
        _window->Show();
    }
    updateWindow();
}

/*
void drwnMatlabNode::hideWindow()
{
    if (_matlabEngine != NULL) {
        engClose(_matlabEngine);
        _matlabEngine = NULL;
    }

    if (_window != NULL) {
        delete _window;
        _window = NULL;
    }
}
*/

void drwnMatlabNode::updateWindow()
{
    if (_matlabEngine == NULL)
        return;
    if ((_window == NULL) || (!_window->IsShown()))
        return;

    // TODO: copy parameters?
}

void drwnMatlabNode::evaluateForwards()
{
    // clear output tables and then update forwards
    clearOutput();
    updateForwards();
}

void drwnMatlabNode::updateForwards()
{
    // error checking
    if (_fwdFcnName.empty()) {
        DRWN_LOG_ERROR("no forward function defined for node \"" << getName() << "\"");
        return;
    }

    if (!drwnFileExists(_fwdFcnName.c_str())) {
        DRWN_LOG_ERROR("can't find forward function \"" << _fwdFcnName << "\" required for node \"" << getName() << "\"");
        return;
    }

    // if no inputs, then execute as source node
    if (_nInputs == 0) {
        sourceUpdateForwards();
        return;
    }

    // check input tables
    for (int i = 0; i < _nInputs; i++) {
        drwnDataTable *tblIn = _inputPorts[i]->getTable();
        if (tblIn == NULL) {
            DRWN_LOG_WARNING("node " << getName() << " requires " << _nInputs << " inputs");
            return;
        }
    }

    // start the Matlab engine if not already open; will be closed on object deletion
    if (_matlabEngine == NULL) {
        if (!(_matlabEngine = engOpen(STARTCMD.c_str()))) {
            DRWN_LOG_ERROR("can't start the Matlab engine with command \"" << STARTCMD << "\"");
            return;
        }
    }

    // change to correct directory
    engEvalString(_matlabEngine,
        (string("cd ") + drwn::strDirectory(_fwdFcnName)).c_str());

    // copy parameters into Matlab
    mxArray *mTheta = eigen2mat(_theta);
    engPutVariable(_matlabEngine, "theta", mTheta);
    mxDestroyArray(mTheta);

    // create Matlab command
    string clrCommand;
    string exeCommand;
    if (_nOutputs > 0) {
        clrCommand = string("clear recOut1");
        exeCommand = string("[recOut1");
        for (int i = 1; i < _nOutputs; i++) {
            clrCommand += string(" recOut") + toString(i + 1);
            exeCommand += string(", recOut") + toString(i + 1);
        }
        clrCommand += string(";");
        exeCommand += string("] = ");
    }
    exeCommand += strBaseName(_fwdFcnName) + string("(");
    for (int i = 0; i < _nInputs; i++) {
        exeCommand += string("recIn") + toString(i + 1) + string(", ");
    }
    exeCommand += string("theta);");

    // interate over input records
    vector<string> keys = _inputPorts[0]->getTable()->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        bool bKeyExists = true;
        for (int i = 0; i < _nOutputs; i++) {
            drwnDataTable *tblOut = _outputPorts[i]->getTable();
            if (!tblOut->hasKey(*it)) {
                bKeyExists = false;
                break;
            }
        }
        if (bKeyExists) continue;

        // export data to Matlab
        for (int i = 0; i < _nInputs; i++) {
            string varName = string("recIn") + toString(i + 1);
            drwnDataTable *tblIn = _inputPorts[i]->getTable();
            drwnDataRecord *recordIn = tblIn->lockRecord(*it);
            mxArray *mRecord = record2mat(recordIn, *it);
            engPutVariable(_matlabEngine, varName.c_str(), mRecord);
            mxDestroyArray(mRecord);
            tblIn->unlockRecord(*it);
        }

        // execute Matlab function
	engEvalString(_matlabEngine, clrCommand.c_str());
        engEvalString(_matlabEngine, exeCommand.c_str());

        // import data from Matlab
        for (int i = 0; i < _nOutputs; i++) {
            string varName = string("recOut") + toString(i + 1);
            mxArray *mRecord = engGetVariable(_matlabEngine, varName.c_str());
            if (mRecord == NULL) {
                DRWN_LOG_ERROR("failed to execute Matlab function \"" << _fwdFcnName << "\" for record \"" << *it << "\"");
                break;
            } else if (!mxIsEmpty(mRecord)) {
                drwnDataTable *tblOut = _outputPorts[i]->getTable();
                drwnDataRecord *recordOut = tblOut->lockRecord(*it);
                string key = mat2record(mRecord, recordOut);
                if (key != *it) {
                    DRWN_LOG_WARNING("Matlab function \"" << _fwdFcnName
                        << "\" changed record key from " << *it << " to " << key);
                }
                DRWN_LOG_DEBUG(_fwdFcnName << " return " << recordOut->numObservations()
                    << "-by-" << recordOut->numFeatures() << " data matrix for key " << *it);
                tblOut->unlockRecord(*it);
            }
            mxDestroyArray(mRecord);
        }
    }
    DRWN_END_PROGRESS;
}

void drwnMatlabNode::sourceUpdateForwards()
{
    // start the Matlab engine
    if (_matlabEngine == NULL) {
        if (!(_matlabEngine = engOpen(STARTCMD.c_str()))) {
            DRWN_LOG_ERROR("can't start the Matlab engine with command \"" << STARTCMD << "\"");
            return;
        }
    }

    // change to correct directory
    engEvalString(_matlabEngine,
        (string("cd ") + drwn::strDirectory(_fwdFcnName)).c_str());

    // copy parameters into Matlab
    mxArray *mTheta = eigen2mat(_theta);
    engPutVariable(_matlabEngine, "theta", mTheta);
    mxDestroyArray(mTheta);

    // create Matlab command
    string clrCommand;
    string exeCommand;
    if (_nOutputs > 0) {
        clrCommand = string("clear recOut1");
        exeCommand = string("[recOut1");
        for (int i = 1; i < _nOutputs; i++) {
            clrCommand += string(" recOut") + toString(i + 1);
            exeCommand += string(", recOut") + toString(i + 1);
        }
        clrCommand += string(";");
        exeCommand += string("] = ");
    }
    exeCommand += strBaseName(_fwdFcnName);

    // iterate until empty record returned
    bool bAllEmpty = false;
    int indx = 0;
    DRWN_START_PROGRESS(getName().c_str(), 100);
    while (!bAllEmpty) {
        if (++indx % 100 == 0) {
            DRWN_START_PROGRESS(getName().c_str(), 100);
        }
        DRWN_INC_PROGRESS;

        // execute Matlab function
	engEvalString(_matlabEngine, clrCommand.c_str());
        engEvalString(_matlabEngine, (exeCommand +
            string("(") + toString(indx) + string(", theta);")).c_str());

        // import data from Matlab
        bAllEmpty = true;
        for (int i = 0; i < _nOutputs; i++) {
            string varName = string("recOut") + toString(i + 1);
            mxArray *mData = engGetVariable(_matlabEngine, varName.c_str());
            if (mData == NULL) {
                DRWN_LOG_ERROR("failed to execute Matlab function \"" << _fwdFcnName << "\" for " << indx << "-th record");
                break;
            } else {
                if (!mxIsEmpty(mData)) {
                    drwnDataRecord *r = new drwnDataRecord();
                    string key = mat2record(mData, r);
                    if (!key.empty()) {
                        drwnDataTable *tblOut = _outputPorts[i]->getTable();
                        tblOut->addRecord(key, r);
                        bAllEmpty = false;
                    } else {
                        delete r;
                    }
                }
                mxDestroyArray(mData);
            }
        }
    }
    DRWN_END_PROGRESS;
}

void drwnMatlabNode::propertyChanged(const string& name)
{
    if ((name == string("numInputs")) || (name == string("numOutputs"))) {
        _nInputs = std::max(0, _nInputs);
        _nOutputs = std::max(0, _nOutputs);
        updatePorts();
    } else {
        drwnNode::propertyChanged(name);
    }
}

void drwnMatlabNode::updatePorts()
{
    // re-assign input ports
    if (_nInputs != (int)_inputPorts.size()) {
        for (vector<drwnInputPort *>::iterator it = _inputPorts.begin();
             it != _inputPorts.end(); it++) {
            delete *it;
        }
        _inputPorts.clear();

        if (_nInputs == 1) {
            _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));
        } else {
            for (int i = 0; i < _nInputs; i++) {
                string portName = string("dataIn") + toString(i);
                _inputPorts.push_back(new drwnInputPort(this, portName.c_str(), "N-by-D data matrix"));
            }
        }
    }

    // re-assign input ports
    if (_nOutputs != (int)_outputPorts.size()) {
        for (vector<drwnOutputPort *>::iterator it = _outputPorts.begin();
             it != _outputPorts.end(); it++) {
            delete *it;
        }
        _outputPorts.clear();

        if (_nOutputs == 1) {
            _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D data matrix"));
        } else {
            for (int i = 0; i < _nOutputs; i++) {
                string portName = string("dataOut") + toString(i);
                _outputPorts.push_back(new drwnOutputPort(this, portName.c_str(), "N-by-D data matrix"));
            }
        }
    }
}

// drwnMATFileSourceNode -----------------------------------------------------

drwnMATFileSourceNode::drwnMATFileSourceNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _bFullRecord(false)
{
    _nVersion = 100;
    _desc = "Imports data from Matlab .mat file";

    // declare ports
    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D data matrix"));

    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("fullRecord", new drwnBooleanProperty(&_bFullRecord));
}

drwnMATFileSourceNode::drwnMATFileSourceNode(const drwnMATFileSourceNode& node) :
    drwnNode(node), _filename(node._filename), _bFullRecord(node._bFullRecord)
{
    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("fullRecord", new drwnBooleanProperty(&_bFullRecord));
}

drwnMATFileSourceNode::~drwnMATFileSourceNode()
{
    // do nothing
}

void drwnMATFileSourceNode::evaluateForwards()
{
    // clear output table and then update forwards
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    tblOut->clear();
    updateForwards();
}

void drwnMATFileSourceNode::updateForwards()
{
    // error checking
    if (!drwnFileExists(_filename.c_str())) {
        DRWN_LOG_ERROR("can't find file \"" << _filename << "\" in node \"" << getName() << "\"");
        return;
    }

    // get output table
    drwnDataTable *tblOut = _outputPorts[0]->getTable();
    DRWN_ASSERT(tblOut != NULL);

    // open .mat file and extract keys
    MATFile *matFile = matOpen(_filename.c_str(), "r");
    if (matFile == NULL) {
        DRWN_LOG_ERROR("could not open \"" + _filename + "\" in node \"" << getName() << "\"");
        return;
    }

    int numRecords = 0;
    char **keys = matGetDir(matFile, &numRecords);
    if (keys == NULL) {
        matClose(matFile);
        DRWN_LOG_WARNING("no records in file \"" << _filename << "\"");
        return;
    }

    mxArray *mData;
    DRWN_START_PROGRESS(getName().c_str(), numRecords);
    for (int i = 0; i < numRecords; i++) {
        DRWN_INC_PROGRESS;
        // don't overwrite existing output records
        if (tblOut->hasKey(keys[i])) continue;

        drwnDataRecord *recordOut = tblOut->lockRecord(keys[i]);
        mData = matGetVariable(matFile, keys[i]);

        // load record
        if (_bFullRecord) {
            string k = mat2record(mData, recordOut);
            DRWN_ASSERT(k == keys[i]);
        } else {
            recordOut->data() = mat2eigen(mData);
        }

        tblOut->unlockRecord(keys[i]);
        mxDestroyArray(mData);
    }
    DRWN_END_PROGRESS;

    // free memory and close file
    mxFree(keys);
    matClose(matFile);
}

// drwnMATFileSinkNode -------------------------------------------------------

drwnMATFileSinkNode::drwnMATFileSinkNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _bFullRecord(false)
{
    _nVersion = 100;
    _desc = "Exports data from Matlab .mat file";

    // declare ports
    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D data matrix"));

    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("fullRecord", new drwnBooleanProperty(&_bFullRecord));
}

drwnMATFileSinkNode::drwnMATFileSinkNode(const drwnMATFileSinkNode& node) :
    drwnNode(node), _filename(node._filename), _bFullRecord(node._bFullRecord)
{
    // declare propertys
    declareProperty("filename", new drwnFilenameProperty(&_filename));
    declareProperty("fullRecord", new drwnBooleanProperty(&_bFullRecord));
}

drwnMATFileSinkNode::~drwnMATFileSinkNode()
{
    // do nothing
}

void drwnMATFileSinkNode::evaluateForwards()
{
    // error checking
    if (_filename.empty()) {
        DRWN_LOG_ERROR("filename is empty in node \"" << getName() << "\"");
        return;
    }

    // get input table
    drwnDataTable *tblIn = _inputPorts[0]->getTable();
    if (tblIn == NULL) {
        DRWN_LOG_WARNING("node " << getName() << " has no input");
        return;
    }

    // open .mat file and extract keys
    MATFile *matFile = matOpen(_filename.c_str(), "w");
    if (matFile == NULL) {
        DRWN_LOG_ERROR("could not open \"" + _filename + "\" for writing in node \"" << getName() << "\"");
        return;
    }

    // interate over input records
    vector<string> keys = tblIn->getKeys();
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;

        // export data
        drwnDataRecord *recordIn = tblIn->lockRecord(*it);

        mxArray *mData = NULL;
        if (_bFullRecord) {
            mData = record2mat(recordIn, *it);
        } else {
            mData = eigen2mat(recordIn->data());
        }
        int status = matPutVariable(matFile, it->c_str(), mData);
	mxDestroyArray(mData);
        if (status != 0) {
            DRWN_LOG_ERROR("failed to write record \"" << *it << "\" to file \"" << _filename << "\"");
        }

        // unlock records
        tblIn->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;

    // close file
    matClose(matFile);
}

// auto registration ---------------------------------------------------------

DRWN_AUTOREGISTERNODE("Custom", drwnMatlabNode);
DRWN_AUTOREGISTERNODE("Source", drwnMATFileSourceNode);
DRWN_AUTOREGISTERNODE("Sink", drwnMATFileSinkNode);


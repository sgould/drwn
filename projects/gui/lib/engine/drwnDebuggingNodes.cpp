/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDebuggingNodes.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"
#include "drwnDebuggingNodes.h"

using namespace std;
using namespace Eigen;

// drwnRandomSourceNode ------------------------------------------------------

drwnRandomSourceNode::drwnRandomSourceNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner), _numRecords(1000), _numFeatures(25),
    _minObservations(1), _maxObservations(1)
{
    _nVersion = 100;
    _desc = "Generates random output";

    _outputPorts.push_back(new drwnOutputPort(this, "dataOut", "N-by-D matrix of data"));

    // declare propertys
    declareProperty("numRecords", new drwnIntegerProperty(&_numRecords));
    declareProperty("numFeatures", new drwnIntegerProperty(&_numFeatures));
    declareProperty("minObservations", new drwnIntegerProperty(&_minObservations));
    declareProperty("maxObservations", new drwnIntegerProperty(&_maxObservations));
}

drwnRandomSourceNode::drwnRandomSourceNode(const drwnRandomSourceNode& node) :
    drwnNode(node), _numRecords(node._numRecords), _numFeatures(node._numFeatures),
    _minObservations(node._minObservations), _maxObservations(node._maxObservations)
{
    // declare propertys
    declareProperty("numRecords", new drwnIntegerProperty(&_numRecords));
    declareProperty("numFeatures", new drwnIntegerProperty(&_numFeatures));
    declareProperty("minObservations", new drwnIntegerProperty(&_minObservations));
    declareProperty("maxObservations", new drwnIntegerProperty(&_maxObservations));
}

drwnRandomSourceNode::~drwnRandomSourceNode()
{
    // do nothing
}

// processing
void drwnRandomSourceNode::evaluateForwards()
{
    drwnDataTable *tbl = _outputPorts[0]->getTable();
    DRWN_ASSERT(tbl != NULL);

    DRWN_ASSERT(_minObservations > 0);
    DRWN_ASSERT(_maxObservations >= _minObservations);
    int obsRange = _maxObservations - _minObservations + 1;

    // generate random output
    DRWN_START_PROGRESS(getName().c_str(), _numRecords);
    for (int i = 0; i < _numRecords; i++) {
        DRWN_INC_PROGRESS;
        string key = string("REC") + drwn::padString(toString(i), 5);
        DRWN_LOG_VERBOSE("generating random record " << key << "...");
        drwnDataRecord *record = tbl->lockRecord(key);
        record->data().setRandom(_minObservations + rand() % obsRange, _numFeatures);
        tbl->unlockRecord(key);
    }
    DRWN_END_PROGRESS;
}

void drwnRandomSourceNode::updateForwards()
{
    drwnDataTable *tbl = _outputPorts[0]->getTable();
    DRWN_ASSERT(tbl != NULL);

    DRWN_ASSERT(_minObservations > 0);
    DRWN_ASSERT(_maxObservations >= _minObservations);
    int obsRange = _maxObservations - _minObservations + 1;

    // generate random output
    DRWN_START_PROGRESS(getName().c_str(), _numRecords);
    for (int i = 0; i < _numRecords; i++) {
        DRWN_INC_PROGRESS;
        string key = string("REC") + drwn::padString(toString(i), 5);

        // don't overwrite existing output records
        if (tbl->hasKey(key)) continue;

        // generate new output record
        DRWN_LOG_VERBOSE("generating random record " << key << "...");
        drwnDataRecord *record = tbl->lockRecord(key);
        record->data().setRandom(_minObservations + rand() % obsRange, _numFeatures);
        tbl->unlockRecord(key);
    }
    DRWN_END_PROGRESS;
}

// drwnStdOutSinkNode --------------------------------------------------------

drwnStdOutSinkNode::drwnStdOutSinkNode(const char *name, drwnGraph *owner) :
    drwnNode(name, owner)
{
    _nVersion = 100;
    _desc = "Writes data to a standard output";

    _inputPorts.push_back(new drwnInputPort(this, "dataIn", "N-by-D matrix of data"));
}

drwnStdOutSinkNode::drwnStdOutSinkNode(const drwnStdOutSinkNode& node) :
    drwnNode(node)
{
    // do nothing
}

drwnStdOutSinkNode::~drwnStdOutSinkNode()
{
    // do nothing
}

// processing
void drwnStdOutSinkNode::evaluateForwards()
{
    drwnDataTable *tbl = _inputPorts[0]->getTable();
    if (tbl == NULL) {
        DRWN_LOG_WARNING("node " << getName() << " has no input");
        return;
    }

    // interate over records
    vector<string> keys = tbl->getKeys();
    DRWN_LOG_VERBOSE("input table " << tbl->name() << " has " << keys.size() << " records...");
    DRWN_START_PROGRESS(getName().c_str(), keys.size());
    for (vector<string>::const_iterator it = keys.begin(); it != keys.end(); it++) {
        DRWN_INC_PROGRESS;
        drwnDataRecord *record = tbl->lockRecord(*it);
        DRWN_ASSERT(record != NULL);
        for (int i = 0; i < record->data().rows(); i++) {
            cout << *it << "\t" << record->data().row(i) << "\n";
        }
        tbl->unlockRecord(*it);
    }
    DRWN_END_PROGRESS;
}


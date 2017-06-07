/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPort.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnDatabase.h"
#include "drwnGraph.h"
#include "drwnNode.h"
#include "drwnPort.h"

using namespace std;
using namespace Eigen;

// drwnDataPort --------------------------------------------------------------

drwnDataPort::drwnDataPort(drwnNode *owner, const char *name, const char *desc) :
    _owner(owner), _name(name), _desc("")
{
    if (desc != NULL) _desc = string(desc);
}

drwnDataPort::drwnDataPort(const drwnDataPort& port) :
    _owner(port._owner), _name(port._name), _desc(port._desc)
{
    // do nothing
}

drwnDataPort::~drwnDataPort()
{
    // do nothing
}

drwnGraph *drwnDataPort::getGraph() const
{
    return (_owner == NULL) ? NULL : _owner->getOwner();
}

// drwnInputPort -------------------------------------------------------------

drwnInputPort::drwnInputPort(drwnNode *owner, const char *name, const char *desc) :
    drwnDataPort(owner, name, desc), _src(NULL), _bRequired(false)
{
    // do nothing
}

drwnInputPort::drwnInputPort(const drwnInputPort& port) :
    drwnDataPort(port), _src(port._src), _bRequired(port._bRequired)
{
    // do nothing
}

drwnInputPort::~drwnInputPort()
{
    // disconnect from source
    disconnect();
}

drwnDataTable *drwnInputPort::getTable() const
{
    return (_src == NULL) ? NULL : _src->getTable();
}

void drwnInputPort::connect(drwnOutputPort *src)
{
    DRWN_ASSERT(src != NULL);

    if (_src == src) return;
    if (_src != NULL) disconnect();
    if (src->getOwner() == _owner) {
        DRWN_LOG_ERROR("cannot connect a node to itself");
    }

    _src = src;
    _src->connect(this);
}

void drwnInputPort::disconnect()
{
    if (_src != NULL) {
        drwnOutputPort *port = _src;
        _src = NULL;
        port->disconnect(this);
    }
}

// drwnOutputPort ------------------------------------------------------------

drwnOutputPort::drwnOutputPort(drwnNode *owner, const char *name, const char *desc) :
    drwnDataPort(owner, name, desc)
{
    // do nothing
}

drwnOutputPort::drwnOutputPort(const drwnOutputPort& port) :
    drwnDataPort(port), _dst(port._dst)
{
    // do nothing
}

drwnOutputPort::~drwnOutputPort()
{
    // disconnect from all destinations
    disconnect();

    // flush database table
    drwnDataTable *table = getTable();
    if (table != NULL) {
        drwnDataCache::get().flush(table);
    }
}

drwnDataTable *drwnOutputPort::getTable() const
{
    drwnGraph *graph = this->getGraph();
    if (graph == NULL) return NULL;

    drwnDatabase *db = graph->getDatabase();
    if (db == NULL) return NULL;

    return db->getTable(_owner->getName() + string(".") + this->getName());
}

void drwnOutputPort::connect(drwnInputPort *dst)
{
    DRWN_ASSERT(dst != NULL);
    if (_dst.find(dst) != _dst.end()) {
        return;
    }

    if (dst->getOwner() == _owner) {
        DRWN_LOG_ERROR("cannot connect a node to itself");
    }

    _dst.insert(dst);
    dst->connect(this);
}

void drwnOutputPort::disconnect(drwnInputPort *dst)
{
    DRWN_ASSERT(dst != NULL);
    set<drwnInputPort *>::iterator it = _dst.find(dst);
    if (it != _dst.end()) {
        _dst.erase(it);
        dst->disconnect();
    }    
}

void drwnOutputPort::disconnect()
{
    vector<drwnInputPort *> ports(_dst.begin(), _dst.end());
    _dst.clear();    
    for (vector<drwnInputPort *>::iterator it = ports.begin(); it != ports.end(); it++) {
        (*it)->disconnect();
    }
}

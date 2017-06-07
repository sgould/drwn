/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPort.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Abstracts input and output connections between nodes.
**
*****************************************************************************/

#pragma once

using namespace std;

// forward declarations ------------------------------------------------------

class drwnGraph;
class drwnNode;

// drwnDataPort --------------------------------------------------------------
// Defines connectivity between nodes. Pure virtual base for drwnInputPort and
// drwnOutputPort.

class drwnDataPort
{
 protected:
    drwnNode *_owner;       // node that owns this port
    string _name;           // name of the port
    string _desc;           // textual description of the port

 public:
    drwnDataPort(drwnNode *owner, const char *name, const char *desc = NULL);
    drwnDataPort(const drwnDataPort& port);
    virtual ~drwnDataPort();

    drwnNode *getOwner() const { return _owner; }
    const char *getName() const { return _name.c_str(); }
    const char *getDescription() const { return _desc.c_str(); }
    void setDescription(const char *desc) { _desc = string(desc); }

    drwnGraph *getGraph() const;
    virtual drwnDataTable *getTable() const = 0;
};

// drwnInputPort -------------------------------------------------------------

class drwnInputPort : public drwnDataPort {
 protected:
    drwnOutputPort *_src;   // port which provides data for this input
    bool _bRequired;        // true if node requires this port for processing

 public:
    drwnInputPort(drwnNode *owner, const char *name, const char *desc = NULL);
    drwnInputPort(const drwnInputPort& port);
    ~drwnInputPort();

    drwnDataTable *getTable() const;
    const drwnOutputPort *getSource() const { return _src; }

    void connect(drwnOutputPort *src);
    void disconnect();
};

// drwnOutputPort ------------------------------------------------------------

class drwnOutputPort : public drwnDataPort {
 protected:
    set<drwnInputPort *> _dst;  // ports which consume data from this output

 public:
    drwnOutputPort(drwnNode *owner, const char *name, const char *desc = NULL);
    drwnOutputPort(const drwnOutputPort& port);
    ~drwnOutputPort();

    drwnDataTable *getTable() const;
    const set<drwnInputPort *>& getTargets() const { return _dst; }

    void connect(drwnInputPort *dst);
    void disconnect(drwnInputPort *dst);
    void disconnect();
};

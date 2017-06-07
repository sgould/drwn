/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNodeFactory.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Factory for creating new nodes. Each node type should register with the
**  factory at startup or when loaded from a DLL or shared library. Nodes are
**  organized by name and (optional) group. The name must be unique. Groups
**  are just used for interfacing. Group id -1 indicates that no group was
**  provided (and these nodes will not appear in the user interfaces).
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnNode.h"

using namespace std;

// drwnNodeFactory -----------------------------------------------------------

class drwnNodeFactory
{
 public:
    typedef drwnNode* (*drwnNodeCreatorFcn)(void);

 protected:
    typedef pair<drwnNodeCreatorFcn, int> drwnNodeFactoryEntry; // (fcn, groupId)
    vector<string> _groups; // ordered groups
    map<string, drwnNodeFactoryEntry> _registry; // indexed by name

 public:
    ~drwnNodeFactory();
    static drwnNodeFactory& get();

    void registerGroup(const char *group);
    void unregisterGroup(const char *group);
    const vector<string> &getGroups() const;

    void registerNode(const char *name, drwnNodeCreatorFcn fcn, const char *group = NULL);
    void unregisterNode(const char *name);
    vector<string> getNodes(const char *group) const;
    vector<string> getNodes(int groupId) const;

    drwnNode *create(const char *name) const;
    drwnNode *create(drwnXMLNode& xml) const;

 protected:
    drwnNodeFactory(); // singleton class so hide constructor
};

// drwnNodeAutoRegister ------------------------------------------------------
// The following macros should be used to automatically register a node with
// the node factory. The first macro should be placed in the class header (.h)
// file and the second in the implementation (.cpp) file.

template<typename T>
class drwnNodeAutoRegister {
 public:
    drwnNodeAutoRegister(const char *group, const char *name) {
        drwnNodeFactory::get().registerNode(name, &drwnNodeAutoRegister<T>::creator, group);
    }

 private:
    static drwnNode *creator() { return new T(); }
};

#ifndef __APPLE__
#define DRWN_DECLARE_AUTOREGISTERNODE(className) \
    template class drwnNodeAutoRegister<className>;
#else
#define DRWN_DECLARE_AUTOREGISTERNODE(className)
#endif

#define DRWN_AUTOREGISTERNODE(group, className) \
    drwnNodeAutoRegister<className> __ ## className ## AutoRegister(group, #className);


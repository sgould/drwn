/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnNodeFactory.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnEngine.h"

#include "drwnNode.h"
#include "drwnNodeFactory.h"

#include "drwnDebuggingNodes.h"

using namespace std;
using namespace Eigen;

// drwnNodeFactory -----------------------------------------------------------

drwnNodeFactory::drwnNodeFactory()
{
    // pre-register groups for correct ordering
    registerGroup("Source");
    registerGroup("Sink");
    registerGroup("Static");
    registerGroup("Adaptive");
    registerGroup("Utility");
    registerGroup("Visualization");
    registerGroup("Custom");
}

drwnNodeFactory::~drwnNodeFactory()
{
    // do nothing
}

drwnNodeFactory& drwnNodeFactory::get()
{
    static drwnNodeFactory factory;
    return factory;
}

void drwnNodeFactory::registerGroup(const char *group)
{
    DRWN_ASSERT(group != NULL);
    DRWN_LOG_DEBUG("registering group \"" << group << "\" with the node factory");
    map<string, drwnNodeFactoryEntry >::iterator it = _registry.find(string(group));
    if (it != _registry.end()) {
        DRWN_LOG_WARNING("group \"" << group << "\" is already registered");
        return;
    }

    _groups.push_back(string(group));
}

void drwnNodeFactory::unregisterGroup(const char *group)
{
    DRWN_ASSERT(group != NULL);
    DRWN_LOG_DEBUG("unregistering group \"" << group << "\" from the node factory");
    vector<string>::iterator it = find(_groups.begin(), _groups.end(), string(group));
    if (it == _groups.end()) {
        DRWN_LOG_WARNING("group \"" << group << "\" does not exist in node registery");
        return;
    }

    int groupId = (int)(it - _groups.begin());
    _groups.erase(it);

    map<string, drwnNodeFactoryEntry>::iterator jt = _registry.begin();
    while (jt != _registry.end()) {
        if (jt->second.second == groupId) {
            map<string, drwnNodeFactoryEntry>::iterator kt = jt++;
            _registry.erase(kt);
            DRWN_LOG_DEBUG("unregistering node \"" << jt->first << "\" from the node factory");
        } else if (jt->second.second >= groupId) {
            jt->second.second -= 1;
            jt++;
        } else {
            jt++;
        }
    }
}

const vector<string> &drwnNodeFactory::getGroups() const
{
    return _groups;
}

void drwnNodeFactory::registerNode(const char *name, drwnNodeCreatorFcn fcn, const char *group)
{
    DRWN_ASSERT((name != NULL) && (fcn != NULL));

    int groupId = -1;
    if (group != NULL) {
        vector<string>::const_iterator it = find(_groups.begin(), _groups.end(), string(group));
        if (it == _groups.end()) {
            DRWN_LOG_DEBUG("registering group \"" << group << "\" with the node factory");
            groupId = (int)_groups.size();
            _groups.push_back(string(group));
        } else {
            groupId = (int)(it - _groups.begin());
        }
    }

    map<string, drwnNodeFactoryEntry>::iterator jt = _registry.find(string(name));
    if (jt != _registry.end()) {
        DRWN_LOG_WARNING("node \"" << name << "\" already registered with node registry");
        jt->second = make_pair(fcn, groupId);
    } else {
        if (group == NULL) {
            DRWN_LOG_DEBUG("registering node \"" << name << "\" with the node factory");
        } else {
            DRWN_LOG_DEBUG("registering node \"" << name << "\" with group \"" << group << "\" in the node factory");
        }
        _registry.insert(make_pair(string(name), make_pair(fcn, groupId)));
    }
}

void drwnNodeFactory::unregisterNode(const char *name)
{
    DRWN_ASSERT(name != NULL);
    DRWN_LOG_DEBUG("unregistering node \"" << name << "\" from the node factory");
    map<string, drwnNodeFactoryEntry>::iterator jt = _registry.find(string(name));
    if (jt == _registry.end()) {
        DRWN_LOG_ERROR("node \"" << name << "\" does not exist in node registery");
        return;
    }

    _registry.erase(jt);
}

vector<string> drwnNodeFactory::getNodes(const char *group) const
{
    DRWN_ASSERT(group != NULL);
    vector<string> names;

    vector<string>::const_iterator it = find(_groups.begin(), _groups.end(), string(group));
    if (it == _groups.end()) {        
        DRWN_LOG_WARNING("group \"" << group << "\" does not exist in node registery");
        return names;
    }
    
    int groupId = (int)(it - _groups.begin());
    return getNodes(groupId);
}

vector<string> drwnNodeFactory::getNodes(int groupId) const
{
    vector<string> names;
    for (map<string, drwnNodeFactoryEntry>::const_iterator jt = _registry.begin();
         jt != _registry.end(); jt++) {
        if (jt->second.second == groupId) {
            names.push_back(jt->first);
        }
    }

    return names;
}

drwnNode *drwnNodeFactory::create(const char *name) const
{
    DRWN_ASSERT(name != NULL);

    map<string, drwnNodeFactoryEntry>::const_iterator jt = _registry.find(string(name));
    if (jt == _registry.end()) {
        DRWN_LOG_ERROR("node \"" << name << "\" does not exist in node registery");
        return NULL;
    }

    return (*jt->second.first)();
}

drwnNode *drwnNodeFactory::create(drwnXMLNode& xml) const
{
    const char *nodeType = drwnGetXMLAttribute(xml, "type");
    if (nodeType == NULL) {
        DRWN_LOG_FATAL("invalid xml node passed to drwnNodeFactory::create()");
        return NULL;
    }

    // construct node and initialize from XML
    drwnNode *node = create(nodeType);
    if (node == NULL) {
        DRWN_LOG_ERROR("node \"" << nodeType << "\" does not exist in node registery");
        return NULL;
    }
    
    node->load(xml);
    return node;
}


// auto registration of known nodes ------------------------------------------
// TODO: move to implementation files
// TODO: allow block description in registration

DRWN_AUTOREGISTERNODE("Source", drwnRandomSourceNode);
DRWN_AUTOREGISTERNODE("Sink", drwnStdOutSinkNode);

// TODO: for testing plugins only (REMOVE)
class drwnLuaDummyNode : public drwnNode {
 public:
    drwnLuaDummyNode(const char *name = NULL, drwnGraph *owner = NULL) :
        drwnNode(name, owner) { }
    drwnLuaDummyNode(const drwnLuaDummyNode& node) : drwnNode(node) { }

    // i/o
    const char *type() const { return "drwnLuaDummyNode"; }
    drwnLuaDummyNode *clone() const { return new drwnLuaDummyNode(*this); }
};
DRWN_AUTOREGISTERNODE("Custom", drwnLuaDummyNode);

// TODO: for testing plugins only (REMOVE)
class drwnPythonDummyNode : public drwnNode {
 public:
    drwnPythonDummyNode(const char *name = NULL, drwnGraph *owner = NULL) :
        drwnNode(name, owner) { }
    drwnPythonDummyNode(const drwnPythonDummyNode& node) : drwnNode(node) { }

    // i/o
    const char *type() const { return "drwnPythonDummyNode"; }
    drwnPythonDummyNode *clone() const { return new drwnPythonDummyNode(*this); }
};
DRWN_AUTOREGISTERNODE("Custom", drwnPythonDummyNode);

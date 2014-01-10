/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGraph.cpp
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

// drwnGraph -----------------------------------------------------------------

const unsigned drwnGraph::_nVersion = 100;

drwnGraph::drwnGraph(const char *title) :
    _database(NULL)
{
    if (title != NULL)
        _title = string(title);
    else _title = string("no title");

    // open an in-memory database
    _database = drwnDbManager::get().openMemoryDatabase();
}

drwnGraph::drwnGraph(const drwnGraph& graph)
{
    DRWN_NOT_IMPLEMENTED_YET;
}

drwnGraph::~drwnGraph()
{
    // close database and delete all nodes
    if (_database != NULL) {
        drwnDbManager::get().closeDatabase(_database);
        _database = NULL;
    }

    for (vector<drwnNode *>::iterator it = _nodes.begin(); it != _nodes.end(); it++) {
        delete *it;
    }
}

// database
void drwnGraph::setDatabase(drwnDatabase *db)
{
    DRWN_ASSERT(db != NULL);
    if (_database != NULL) {
        drwnDbManager::get().closeDatabase(_database);
    }
    _database = db;
}

void drwnGraph::setDatabase(const char *dbName)
{
    if (_database != NULL) {
        drwnDbManager::get().closeDatabase(_database);
    }
    if (dbName == NULL) {
        _database = drwnDbManager::get().openMemoryDatabase();
    } else {
        _database = drwnDbManager::get().openDatabase(dbName);
        if (_database == NULL) {
            DRWN_LOG_VERBOSE("opening in-memory database");
            _database = drwnDbManager::get().openMemoryDatabase();
        }
    }
}

// get the next available node name
string drwnGraph::getNewName(const char *baseName) const
{
    int baseLength = strlen(baseName);
    string baseNamePlus = string(baseName) + string(" (");

    int freeIndex = 0;
    for (vector<drwnNode *>::const_iterator it = _nodes.begin(); it != _nodes.end(); it++) {
        if ((freeIndex == 0) && ((*it)->getName().compare(0, baseLength, baseName) == 0)) {
            freeIndex = 1;
        }

        if ((*it)->getName().compare(0, baseNamePlus.size(), baseNamePlus) == 0) {
            int indx = atoi((*it)->getName().substr(baseNamePlus.size(),
                (*it)->getName().length() - baseNamePlus.size() - 1).c_str());
            if (indx >= freeIndex)
                freeIndex = indx + 1;
        }
    }

    if (freeIndex == 0) {
        return string(baseName);
    }

    return baseNamePlus + toString(freeIndex) + string(")");
}

// adding, deleting and retrieving nodes
bool drwnGraph::addNode(drwnNode *node)
{
    DRWN_ASSERT((node != NULL) && (node->getOwner() == NULL));

    if (node->getName().empty()) {
        node->setName(getNewName().c_str());
    }

    node->setOwner(this);
    _nodes.push_back(node);

    return true;
}

bool drwnGraph::delNode(const char *name)
{
    DRWN_ASSERT(name != NULL);

    int nodeIndx = findNode(name);
    if (nodeIndx < 0) {
        DRWN_LOG_ERROR("can't find node " << name << " in graph " << _title);
        return false;
    }

    return delNode(_nodes[nodeIndx]);
}

bool drwnGraph::delNode(drwnNode *node)
{
    DRWN_ASSERT(node != NULL);

    vector<drwnNode *>::iterator it = find(_nodes.begin(), _nodes.end(), node);
    DRWN_ASSERT(it != _nodes.end());

    // delete records associated with the node
    for (int i = 0; i < (*it)->numOutputPorts(); i++) {
        drwnDataTable *table = (*it)->getOutputPort(i)->getTable();
        if (table != NULL) {
            table->getOwner()->deleteTable(table->name());
        }
    }

    // delete node
    (*it)->setOwner(NULL);
    delete *it;
    _nodes.erase(it);

    return true;
}

bool drwnGraph::delNode(int indx)
{
    DRWN_ASSERT((unsigned)indx < _nodes.size());
    return delNode(_nodes[indx]);
}

int drwnGraph::findNode(const char *name) const
{
    vector<drwnNode *>::const_reverse_iterator it = _nodes.rbegin();
    int indx = (int)_nodes.size() - 1;
    while (it != _nodes.rend()) {
        if ((*it)->getName() == string(name))
            break;
        it++; indx -= 1;
    }

    return indx;
}

drwnNode *drwnGraph::getNode(const char *name) const
{
    int nodeIndx = findNode(name);
    if (nodeIndx < 0) {
        DRWN_LOG_ERROR("can't find node " << name << " in graph " << _title);
        return NULL;
    }

    return _nodes[nodeIndx];
}

drwnNode *drwnGraph::getNode(int indx) const
{
    DRWN_ASSERT((indx >= 0) && (indx < (int)_nodes.size()));
    return _nodes[indx];
}

set<drwnNode *> drwnGraph::copySubGraph(const set<drwnNode *>& nodes) const
{
    set<drwnNode *> copiedNodes;

    // copy nodes
    map<drwnNode *, drwnNode *> nodeMapping;
    for (set<drwnNode *>::const_iterator it = nodes.begin(); it != nodes.end(); it++) {
        drwnNode *n = static_cast<drwnNode *>((*it)->clone());
        n->setOwner(NULL);
        copiedNodes.insert(n);
        nodeMapping[*it] = n;
    }

    // copy connections
    for (set<drwnNode *>::const_iterator it = nodes.begin(); it != nodes.end(); it++) {
        for (int i = 0; i < (*it)->numInputPorts(); i++) {
            const drwnOutputPort *port = (*it)->getInputPort(i)->getSource();
            if ((port != NULL) && (nodes.find(port->getOwner()) != nodes.end())) {
                drwnNode *dst = nodeMapping[*it];
                drwnNode *src = nodeMapping[port->getOwner()];
                dst->getInputPort(i)->connect(src->getOutputPort(port->getName()));
            }
        }
    }

    return copiedNodes;
}

bool drwnGraph::pasteSubGraph(const set<drwnNode *>& nodes, int x, int y)
{
    // determine paste offset
    int minX = DRWN_INT_MAX;
    int minY = DRWN_INT_MAX;
    for (set<drwnNode *>::const_iterator it = nodes.begin(); it != nodes.end(); it++) {
        minX = std::min(minX, (*it)->getLocationX());
        minY = std::min(minY, (*it)->getLocationY());
    }

    // paste nodes
    map<drwnNode *, drwnNode *> nodeMapping;
    for (set<drwnNode *>::const_iterator it = nodes.begin(); it != nodes.end(); it++) {
        drwnNode *n = static_cast<drwnNode *>((*it)->clone());
        n->setOwner(this);
        n->setName(getNewName(n->getName().c_str()).c_str());
        n->setLocation(n->getLocationX() - minX + x, n->getLocationY() - minY + y);

        _nodes.push_back(n);
        nodeMapping[*it] = n;
    }

    // paste connections
    for (set<drwnNode *>::const_iterator it = nodes.begin(); it != nodes.end(); it++) {
        for (int i = 0; i < (*it)->numInputPorts(); i++) {
            const drwnOutputPort *port = (*it)->getInputPort(i)->getSource();
            if ((port != NULL) && (nodes.find(port->getOwner()) != nodes.end())) {
                drwnNode *dst = nodeMapping[*it];
                drwnNode *src = nodeMapping[port->getOwner()];
                dst->getInputPort(i)->connect(src->getOutputPort(port->getName()));
            }
        }
    }

    return true;
}

bool drwnGraph::connectNodes(drwnNode *srcNode, const char *srcPort,
    drwnNode *dstNode, const char *dstPort)
{
    DRWN_ASSERT((srcNode != NULL) && (srcPort != NULL));
    DRWN_ASSERT((dstNode != NULL) && (dstPort != NULL));

    drwnOutputPort *outPort = srcNode->getOutputPort(srcPort);
    drwnInputPort *inPort = dstNode->getInputPort(dstPort);

    inPort->connect(outPort);
    return true;
}

bool drwnGraph::connectNodes(drwnNode *srcNode, int srcPort,
    drwnNode *dstNode, int dstPort)
{
    DRWN_ASSERT((srcNode != NULL) && (dstNode != NULL));

    drwnOutputPort *outPort = srcNode->getOutputPort(srcPort);
    drwnInputPort *inPort = dstNode->getInputPort(dstPort);

    inPort->connect(outPort);
    return true;
}

// file i/o
void drwnGraph::clear()
{
    _title = string("no title");
    _notes.clear();

    for (vector<drwnNode *>::iterator it = _nodes.begin(); it != _nodes.end(); it++) {
        delete *it;
    }
    _nodes.clear();

    setDatabase((const char *)NULL);
}

bool drwnGraph::load(drwnXMLNode& xml)
{
    DRWN_ASSERT(!drwnIsXMLEmpty(xml));

    // delete existing nodes (and close database)
    clear();

    // check version number
    if (drwnGetXMLAttribute(xml, "version") != NULL) {
        unsigned v = atoi(drwnGetXMLAttribute(xml, "version"));
        if (v > _nVersion) {
            DRWN_LOG_WARNING("graph has higher version (" << v << ") than recognized by this build");
        }
    }

    // read title
    DRWN_ASSERT(drwnGetXMLAttribute(xml, "title") != NULL);
    _title = string(drwnGetXMLAttribute(xml, "title"));

    // read database
    if (drwnGetXMLAttribute(xml, "db") != NULL) {
        setDatabase(drwnGetXMLAttribute(xml, "db"));
    }

    // read nodes
    for (drwnXMLNode *child = xml.first_node("drwnNode"); child != NULL;
         child = child->next_sibling("drwnNode")) {
        drwnNode *node = drwnNodeFactory::get().create(*child);
        if (node == NULL) {
            DRWN_LOG_ERROR("node may have been created by a later version or missing plugin.");
            continue;
        }
        addNode(node);
    }

    // read ports (connections)
    for (drwnXMLNode *child = xml.first_node("drwnPort"); child != NULL;
         child = child->next_sibling("drwnPort")) {

        // error checking
        drwnNode *srcNode = getNode(drwnGetXMLAttribute(*child, "srcNode"));
        if (srcNode == NULL) {
            DRWN_LOG_ERROR("node \"" << drwnGetXMLAttribute(*child, "srcNode")
                << "\" does not exist in the network and cannot be connected");
            continue;
        }
        drwnOutputPort *srcPort = srcNode->getOutputPort(drwnGetXMLAttribute(*child, "srcPort"));
        if (srcPort == NULL) {
            DRWN_LOG_ERROR("node \"" << srcNode->getName() << "\" has no port named \""
                << drwnGetXMLAttribute(*child, "srcPort"));
            continue;
        }
        drwnNode *dstNode = getNode(drwnGetXMLAttribute(*child, "dstNode"));
        if (dstNode == NULL) {
            DRWN_LOG_ERROR("node \"" << drwnGetXMLAttribute(*child, "dstNode")
                << "\" does not exist in the network and cannot be connected");
            continue;
        }
        drwnInputPort *dstPort = dstNode->getInputPort(drwnGetXMLAttribute(*child, "dstPort"));
        if (dstPort == NULL) {
            DRWN_LOG_ERROR("node \"" << dstNode->getName() << "\" has no port named \""
                << drwnGetXMLAttribute(*child, "dstPort"));
            continue;
        }

        // connection
        srcPort->connect(dstPort);
    }

    return true;
}

bool drwnGraph::save(drwnXMLNode& xml) const
{
    // header
    drwnAddXMLAttribute(xml, "title", _title.c_str(), false);
    drwnAddXMLAttribute(xml, "version", toString(_nVersion).c_str(), false);

    // database
    if (_database->isPersistent()) {
        drwnAddXMLAttribute(xml, "db", _database->name().c_str(), false);
    }

    // nodes
    for (vector<drwnNode *>::const_iterator it = _nodes.begin(); it != _nodes.end(); it++) {
        DRWN_ASSERT((*it) != NULL);
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "drwnNode", NULL, false);
        (*it)->save(*node);
    }

    // ports (connections)
    for (vector<drwnNode *>::const_iterator it = _nodes.begin(); it != _nodes.end(); it++) {
        for (int i = 0; i < (*it)->numInputPorts(); i++) {
            drwnInputPort *port = (*it)->getInputPort(i);
            if (port->getSource() == NULL) continue;
            drwnXMLNode *portNode = drwnAddXMLChildNode(xml, "drwnPort", NULL, false);
            drwnAddXMLAttribute(*portNode, "srcNode",
                port->getSource()->getOwner()->getName().c_str(), false);
            drwnAddXMLAttribute(*portNode, "srcPort", port->getSource()->getName(), false);
            drwnAddXMLAttribute(*portNode, "dstNode", (*it)->getName().c_str(), false);
            drwnAddXMLAttribute(*portNode, "dstPort", port->getName(), false);
        }
    }

    return true;
}

// evaluation
void drwnGraph::evaluateForwards(const set<drwnNode *>& nodeSet)
{
    // TODO: order just needs to be topological over nodeSet
    list<drwnNode *> order;
    if (!topologicalSortNodes(order)) {
        DRWN_LOG_ERROR("could not evaluate due to loops in graph");
        return;
    }

    drwnLogger::setRunning(true);
    for (list<drwnNode *>::iterator it = order.begin(); it != order.end(); it++) {
        if (nodeSet.empty() || (nodeSet.find(*it) != nodeSet.end())) {
            (*it)->initializeForwards();
            (*it)->evaluateForwards();
            (*it)->finalizeForwards();
        }
        if (!drwnLogger::isRunning()) break;
    }
    drwnLogger::setRunning(false);
}

void drwnGraph::updateForwards(const set<drwnNode *>& nodeSet)
{
    list<drwnNode *> order;
    if (!topologicalSortNodes(order)) {
        DRWN_LOG_ERROR("could not update due to loops in graph");
        return;
    }

    drwnLogger::setRunning(true);
    for (list<drwnNode *>::iterator it = order.begin(); it != order.end(); it++) {
        if (nodeSet.empty() || (nodeSet.find(*it) != nodeSet.end())) {
            (*it)->initializeForwards(false);
            (*it)->updateForwards();
            (*it)->finalizeForwards();
        }
        if (!drwnLogger::isRunning()) break;
    }
    drwnLogger::setRunning(false);
}

void drwnGraph::propagateBackwards(const set<drwnNode *>& nodeSet)
{
    DRWN_NOT_IMPLEMENTED_YET;
}

void drwnGraph::resetParameters(const set<drwnNode *>& nodeSet)
{
    // reset parameters
    if (nodeSet.empty()) {
        for (vector<drwnNode *>::const_iterator it = _nodes.begin();
             it != _nodes.end(); it++) {
            (*it)->resetParameters();
        }
    } else {
        for (set<drwnNode *>::const_iterator it = nodeSet.begin();
             it != nodeSet.end(); it++) {
            (*it)->resetParameters();
        }
    }
}

void drwnGraph::initializeParameters(const set<drwnNode *>& nodeSet)
{
    list<drwnNode *> order;
    if (!topologicalSortNodes(order)) {
        DRWN_LOG_ERROR("could not update due to loops in graph");
        return;
    }

    // TODO: need to update forwards all predecessors for nodeSet
    drwnLogger::setRunning(true);
    for (list<drwnNode *>::iterator it = order.begin(); it != order.end(); it++) {
        if (nodeSet.empty() || (nodeSet.find(*it) != nodeSet.end())) {
            // clear back-propagated gradients
            (*it)->clearOutput();

            // initialize parameters
            (*it)->initializeParameters();

            // make data available to upstream nodes
            (*it)->initializeForwards();
            (*it)->evaluateForwards();
            (*it)->finalizeForwards();
        }
        if (!drwnLogger::isRunning()) break;
    }
    drwnLogger::setRunning(false);
}

// Sort nodes into a depth-first topological order for forward
// evaluation and gradient back-propagation.
bool drwnGraph::topologicalSortNodes(list<drwnNode *>& order)
{
    order.clear();
    set<drwnNode *> toBeQueued(_nodes.begin(), _nodes.end());
    set<drwnNode *> visited;

    // find initial nodes
    deque<drwnNode *> frontier;
    for (set<drwnNode *>::iterator it = toBeQueued.begin(); it != toBeQueued.end(); ) {
        set<drwnNode *>::iterator jt = it++;
        if ((*jt)->numInputPorts() == 0) {
            frontier.push_back(*jt);
            toBeQueued.erase(jt);
        }
    }

    // keep adding nodes with sources already added
    DRWN_LOG_DEBUG("Sorting nodes...");
    while (!frontier.empty()) {
        drwnNode *nextNode = frontier.front();
        frontier.pop_front();
        visited.insert(nextNode);
        order.push_back(nextNode);

        DRWN_LOG_DEBUG("...adding node \"" << nextNode->getName() << "\"");

        for (set<drwnNode *>::iterator it = toBeQueued.begin();
             it != toBeQueued.end(); ) {
            set<drwnNode *>::iterator jt = it++;

            bool bEnqueue = true;
            for (int j = 0; j < (*jt)->numInputPorts(); j++) {
                drwnInputPort *p = (*jt)->getInputPort(j);
                if (p->getSource() == NULL) continue;
                drwnNode *n = p->getSource()->getOwner();
                if (visited.find(n) == visited.end()) {
                    bEnqueue = false;
                    break;
                }
            }

            if (bEnqueue) {
                frontier.push_front(*jt);
                toBeQueued.erase(*jt);
            }
        }
    }

    // check for loops
    if (order.size() != _nodes.size()) {
        string str;
        for (set<drwnNode *>::iterator it = toBeQueued.begin(); it != toBeQueued.end(); it++) {
            str += string(" \"") + (*it)->getName() + string("\"");
        }
        DRWN_LOG_ERROR("Network contains loops involving nodes:" << str);
        return false;
    }

    return true;
}

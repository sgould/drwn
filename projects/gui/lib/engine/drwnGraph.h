/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGraph.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Defines the data flow graph which holds data processing nodes.
**
*****************************************************************************/

#pragma once

using namespace std;

// forward declarations ------------------------------------------------------

class drwnNode;
class drwnInputPort;
class drwnOutputPort;

// drwnGraph -----------------------------------------------------------------

class drwnGraph : public drwnStdObjIface {
 protected:
    static const unsigned _nVersion;   // version number for forward compatibility 

    string _title;
    string _notes;

    vector<drwnNode *> _nodes; // TODO: change to list?
    drwnDatabase *_database;

 public:
    drwnGraph(const char *title = NULL);
    drwnGraph(const drwnGraph& graph);
    virtual ~drwnGraph();

    const string& getTitle() const { return _title; }
    void setTitle(const string& title) { _title = title; }
    const string& getNotes() const { return _notes; }
    void setNotes(const string& notes) { _notes = notes; }

    // database (the graph is responsible for closing when
    // the database changes or the graph is destroyed)
    drwnDatabase *getDatabase() const { return _database; }
    void setDatabase(drwnDatabase *db);
    void setDatabase(const char *dbName = NULL);
    
    // number of nodes in the graph
    int numNodes() const { return (int)_nodes.size(); }
    
    // get name of the form "new node (<i>)"
    string getNewName(const char *baseName = "new node") const;

    // adding, deleting and retrieving nodes
    bool addNode(drwnNode *node);
    bool delNode(const char *name);
    bool delNode(drwnNode *node);
    bool delNode(int indx);

    int findNode(const char *name) const;
    drwnNode *getNode(const char *name) const;
    drwnNode *getNode(int indx) const;

    // copy and paste subgraphs
    set<drwnNode *> copySubGraph(const set<drwnNode *>& nodes) const;
    bool pasteSubGraph(const set<drwnNode *>& nodes, int x = 0, int y = 0);

    // connectivity
    bool connectNodes(drwnNode *srcNode, const char *srcPort, drwnNode *dstNode, const char *dstPort);
    bool connectNodes(drwnNode *srcNode, int srcPort, drwnNode *dstNode, int dstPort);
    bool disconnectNode(drwnNode *node);
    bool disconnectInput(drwnNode *node, const char *inPort);
    bool disconnectOutput(drwnNode *node, const char *outPort);

    // i/o
    const char *type() const { return "drwnGraph"; }
    drwnGraph *clone() const { return new drwnGraph(*this); }
    void clear();
    bool load(drwnXMLNode& xml);
    bool save(drwnXMLNode& xml) const;

    // Evaluate set of nodes (in topological order). If node set
    // is empty, then evaluate all nodes.
    void evaluateForwards(const set<drwnNode *>& nodeSet = set<drwnNode *>());
    void updateForwards(const set<drwnNode *>& nodeSet = set<drwnNode *>());
    void propagateBackwards(const set<drwnNode *>& nodeSet = set<drwnNode *>());
    void resetParameters(const set<drwnNode *>& nodeSet = set<drwnNode *>());
    void initializeParameters(const set<drwnNode *>& nodeSet = set<drwnNode *>());

    // TODO: global optimization

 protected:
    // Sort nodes in a topological order for forward evaluation and
    // gradient backpropagation. Returns false if loops are detected.
    bool topologicalSortNodes(list<drwnNode *>& order);

};

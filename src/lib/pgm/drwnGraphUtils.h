/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGraphUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Generic graph utilities.
**
*****************************************************************************/

/*!
** \file drwnGraphUtils.h
** \anchor drwnGraphUtils
** \brief Generic graph utilities.
*/

#pragma once

#include <cassert>
#include <vector>
#include <map>
#include <set>

using namespace std;

// basic data types ---------------------------------------------------------

//! directed or undirected edges
typedef std::pair<int, int> drwnEdge;
ostream& operator<<(ostream& os, const drwnEdge& e);

//! variable clique
typedef std::set<int> drwnClique;

class drwnWeightedEdge {
 public:
    int nodeA;    //!< source node
    int nodeB;    //!< target node
    double wAB;   //!< weight from source to target
    double wBA;   //!< weight from target to source

 public:
    drwnWeightedEdge() : 
        nodeA(-1), nodeB(-1), wAB(0.0), wBA(0.0) { /* do nothing */ };
    drwnWeightedEdge(int a, int b, double w = 0.0, double v = 0.0) :
        nodeA(a), nodeB(b), wAB(w), wBA(v) { /* do nothing */ };
    ~drwnWeightedEdge() { /* do nothing */ };
};

typedef enum _drwnTriangulationHeuristic {
    DRWN_MAXCARDSEARCH, //!< maximum cardinality search (MCS)
    DRWN_MINFILLIN      //!< minimum fill-in heuristic
} drwnTriangulationHeuristic;

// graph utilities ----------------------------------------------------------

//! Finds the neighbors for each node in a graph.
vector<drwnClique> findNeighbors(const vector<drwnEdge>& graph);

//! Find a clique containing the subclique
int findSuperset(const vector<drwnClique>& cliques, const drwnClique& subclique);

//! Finds the minimum weight spanning tree (forest) given a graph structure
//! and weights for each edge (missing edges are assumed to have infinite
//! weight). Implementation is based on Kruskal's algorithm.
vector<drwnEdge> minSpanningTree(int numNodes, const vector<drwnEdge>& edges,
    const vector<double>& weights);

//! Maximum cardinality search. Returns an elimination ordering for a chordal
//! graph (true) or three nodes that need to be triangulated for non-chordal
//! graphs (return value false). The user can supply the starting node for the
//! search.
bool maxCardinalitySearch(int numNodes, const vector<drwnEdge>& edges,
    vector<int>& perfectOrder, int startNode = -1);

//! Triangulate an undirected graph. Modifies the adjacency list inline. The
//! resulting graph contains no cycles of length four or more without a chord.
//! Different triangulation methods are supported.
void triangulateGraph(int numNodes, vector<drwnEdge>& edges,
    drwnTriangulationHeuristic method = DRWN_MINFILLIN);

//! Finds cliques obtained by variable elimination.
vector<drwnClique> variableEliminationCliques(const vector<drwnEdge>& edges,
    const vector<int>& nodeOrder);

//! Floyd-Warshall algorithm for all paths shortest path
vector<vector<double> > allShortestPaths(int numNodes, const vector<drwnEdge>& edges,
    const vector<double>& weights);

//! Dijkstra's shortest path algorithm
vector<int> shortestPath(int numNodes, const vector<drwnEdge>& edges,
    const vector<double>& weights, int source, int sink);

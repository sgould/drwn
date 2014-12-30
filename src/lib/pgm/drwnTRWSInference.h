/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTRWSInference.h
** AUTHOR(S):   Hendra Gunadi <u4971560@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnFactorGraph.h"
#include "drwnMapInference.h"

using namespace std;

// drwnTRWSInference -----------------------------------------------------------
//! Implements the sequential tree-reweighted message passing (TRW-S) algorithm
//! described in "Convergent Tree-Reweighted Message Passing for Energy
//! Minimization," Kolmogorov, IEEE PAMI, 2006.

class drwnTRWSInference : public drwnMAPInference
{
 public:
    static int CONVERGENCE_STEP;            //!< number of steps checked in determining convergence
    static double EPSILON;                  //!< Used to define the treshold for stopping condition
    static double THETA_CONST;              //!< define the constant

 private:
    int _numNodes;
    vector<vector<int> > _next1;                                // Populate the sucessors in increasing direction
    vector<vector<int> > _prev1;                                // Populate the predecessors in increasing direction
    vector<vector<int> > _next2;                                // Populate the sucessors in decreasing direction
    vector<vector<int> > _prev2;                                // Populate the predecessors in decreasing direction
    vector<vector<const drwnTableFactor*> > _outFactor1;        // Populate the index of table factor for the outgoing edges (inc)
    vector<vector<const drwnTableFactor*> > _outFactor2;        // Populate the index of table factor for the outgoing edges (dec)
    vector<vector<drwnTableFactor*> > _tempOutFactor1;          // Populate the index of temporary table factor for the outgoing edges (inc)
    vector<vector<drwnTableFactor*> > _tempOutFactor2;          // Populate the index of temporary table factor for the outgoing edges (dec)
    vector<vector<const drwnTableFactor*> > _inFactor1;         // Populate the index of table factor for the incoming edges (inc)
    vector<vector<const drwnTableFactor*> > _inFactor2;         // Populate the index of table factor for the incoming edges (dec)
    vector<const drwnTableFactor*> _unary;                      // Populate the unary node
    vector<vector<drwnTableFactor*> > _outMsg1;                 // Messages for sucessor(s) in increasing direction
    vector<vector<drwnTableFactor*> > _inPrevMsg1;              // Messages for predecessor(s) appear before the node in increasing direction
    vector<vector<drwnTableFactor*> > _inMsg1;                  // Populate all the incoming messages in increasing direction
    vector<vector<drwnTableFactor*> > _outMsg2;                 // Messages for sucessor(s) in decreasing direction
    vector<vector<drwnTableFactor*> > _inPrevMsg2;              // Messages for predecessor(s) appear before the node in decreasing direction
    vector<vector<drwnTableFactor*> > _inMsg2;                  // Populate all the incoming messages in decreasing direction
    vector<drwnTableFactor*> _calibrated;                       // Calibrated Unary Table Factor
    vector<set<int> > _margin;                                  // The set of variables to minimize over
    vector<bool> _flag;                                         // Populate the added unary
    vector<vector<drwnFactorOperation*> > _incUnaryOp;          // Unary factor operations (inc) mapping to be executed later
    vector<vector<drwnFactorOperation*> > _decUnaryOp;          // Unary factor operations (dec) mapping to be executed later
    vector<vector<vector<drwnFactorOperation*> > > _incMsgOp;   // Outgoing message factor operations (inc) mapping to be executed later
    vector<vector<vector<drwnFactorOperation*> > > _decMsgOp;   // Outgoing message factor operations (dec) mapping to be executed later
    bool _initialized;                                          // Indicate whether the object has been initialized
    
 public:
    drwnTRWSInference(const drwnFactorGraph& graph);
    ~drwnTRWSInference();
    
    void clear();
    pair<double, double> inference(drwnFullAssignment& mapAssignment);

 private:
    void initialize();
    void buildComputationGraph();
};

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
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

// drwnADLPInference -----------------------------------------------------------
//! Implements the alternating direction method algorithm described in
//! "An Alternating Direction Method for Dual MAP LP Relaxation," Ofer Meshi and
//! Amir Globerson, ECML, 2011.

class drwnADLPInference : public drwnMAPInference
{
 public:
    static int MAX_ITERATIONS;                                          //!< maximum number of iterations
    static double EPSILON;                                              //!< Used to define the treshold for stopping condition
    static double PENALTY_PARAMETER;                                    //!< Used to define initial Rho

 private:
    int _numNodes;                                                      // Total number of variables involved
    int _cliqueSize;                                                    // Total number of clique with more than 1 variables involved
    vector<const drwnTableFactor* > _unary;                             // Populate the unary variables
    vector<drwnTableFactor* > _unary_bar;                               // Temporary tables used to store immediate result (especially in update delta)
    vector<const drwnTableFactor* > _clique;                            // Populate the cliques
    vector<drwnTableFactor* > _clique_bar;                              // Temporary tables used to store immediate result (especially in update lambda)
    vector<vector<drwnTableFactor* > > _message_unary;                  // Populate the tables to store messages from unary variables to clique(s) -> delta
    vector<vector<drwnTableFactor* > > _message_unary_bar;              // Populate the tables to store calibrated messages from unary variables to clique(s) -> delta bar
    vector<vector<drwnTableFactor* > > _message_clique;                 // Populate the messages received by each clique (the same entries as the message_unary)
    vector<vector<drwnTableFactor* > > _message_clique_bar;             // Populate the calibrated messages received by each clique (the same entries as the message_unary_bar)
    vector<vector<drwnTableFactor* > > _gamma;                          // Populate the dual parameter gamma for each message from unary
    vector<vector<drwnTableFactor* > > _gamma_clique;                   // Populate the dual parameter gamma sent for each clique
    vector<drwnTableFactor* > _mu;                                      // Populate the dual parameter mu for each clique
    vector<drwnTableFactor* > _tempMu;                                  // Temporary tables used to store immediate result (especially in update mu)
    vector<drwnTableFactor* > _lambda;                                  // Populate the parameter lambda for each clique
    vector<vector<set<int> > > _marginalizer;                           // Sets of variables used to marginalize (used in update delta bar) 
    vector<vector<drwnTableFactor*> > _margin_result_lambda;            // Temporary table to store marginalized lambda (in updating delta bar operation)
    vector<vector<drwnTableFactor*> > _margin_result_mu;                // Temporary table to store marginalized mu (in updating delta bar operation)
    vector<bool> _flag;                                                 // Populate the added unary
    vector<vector<drwnFactorOperation*> > _updateLambdaOp;              // Pre-computed computation graph required for updating lambda
    vector<vector<vector<drwnFactorOperation*> > > _updateDeltaBarOp;   // Pre-computed computation graph required for updating delta bar
    vector<vector<drwnFactorOperation*> > _updateMuOp;                  // Pre-computed computation graph required for updating mu
    vector<vector<drwnFactorOperation*> > _decodeOp;                    // Pre-computed computation graph required for decoding assignment

 public:
    drwnADLPInference(const drwnFactorGraph& graph);
    ~drwnADLPInference();

    void clear();
    pair<double, double> inference(drwnFullAssignment& mapAssignment);

 private:
    double TRIM(drwnTableFactor* v, double z);                          // Internal functions to implement the TRIM function described in the paper
    void buildComputationGraph();                                       // Construct pre-computed computation graphs
};

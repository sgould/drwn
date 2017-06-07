/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnRollupNode.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Computes summary statistics over disjoint sets of observations. Index
**  input should provide a matrix with the same number of rows as the input.
**  Each entry represents an index into the output matrix (and should be
**  numbered contiguously from 0). If the index input is missing then the
**  entire input is summarized.
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnEngine.h"

using namespace std;

// drwnRollupNode ------------------------------------------------------------

class drwnRollupNode : public drwnNode {
 protected:
    static vector<string> _operations;
    int _selection;

 public:
    drwnRollupNode(const char *name = NULL, drwnGraph *owner = NULL);
    drwnRollupNode(const drwnRollupNode& node);
    virtual ~drwnRollupNode();

    // i/o
    const char *type() const { return "drwnRollupNode"; }
    drwnRollupNode *clone() const { return new drwnRollupNode(*this); }

    // processing
    void evaluateForwards();
    void updateForwards();    
    void propagateBackwards();

 protected:
    MatrixXd fcnMean(const MatrixXd& data, const set<int>& index) const;
    MatrixXd fcnVariance(const MatrixXd& data, const set<int>& index) const;
    MatrixXd fcnStdev(const MatrixXd& data, const set<int>& index) const;

    //MatrixXd gradMean(const MatrixXd& data, const set<int>& index) const;
    //MatrixXd gradVariance(const MatrixXd& data, const set<int>& index) const;   
};

// auto registration ---------------------------------------------------------

DRWN_DECLARE_AUTOREGISTERNODE(drwnRollupNode);


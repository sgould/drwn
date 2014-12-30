/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnInference.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <set>
#include <map>

#include "drwnBase.h"
#include "drwnVarUniverse.h"
#include "drwnVarAssignment.h"
#include "drwnFactorGraph.h"
#include "drwnTableFactorOps.h"

// drwnInference class -----------------------------------------------------
//! Interface for various (marginal) inference algorithms.
//!
//! Algorithms operate on a factor graph (see drwnFactorGraph) that is
//! assumed to be provided as unnormalized probability distributions, i.e.,
//! \f$P(x) \propto \prod_c \Psi_c(x_c)\f$ and the task is to find the
//! marginal distibution over each variable.
//! A constant reference is maintained to the factor graph, so it is
//! important that the drwnFactorGraph object not be destoryed before
//! destroying the drwnMapInference object.
//! \todo derive from drwnStdObjectIface to allow factory creation

class drwnInference {
 protected:
    const drwnFactorGraph& _graph; //! reference to initial clique potentials

 public:
    drwnInference(const drwnFactorGraph& graph);
    drwnInference(const drwnInference& inf);
    virtual ~drwnInference();

    //! clear internally cached data (e.g., computation graph)
    virtual void clear() { /* do nothing */ };
    //! run inference (or resume for iterative algorithms) and
    //! return true if converged
    virtual bool inference() = 0;
    //! return the belief over the variables in the given factor, which
    //! must be one of the cliques in the original factor graph
    virtual void marginal(drwnTableFactor& belief) const = 0;
    //! returns marginals for each variable in the factor graph's universe
    virtual drwnFactorGraph varMarginals() const;

    //! return the marginal distribution over variable \p varIndx
    inline drwnTableFactor operator[](int varIndx) const {
        drwnTableFactor factor(_graph.getUniverse());
        factor.addVariable(varIndx);
        marginal(factor);
        return factor;
    }
    //! return the marginal distribution over variable \p varName
    inline drwnTableFactor operator[](const char* varName) const {
        return (*this)[_graph.getUniverse()->findVariable(varName)];
    }
};

// drwnMessagePassingInference class ----------------------------------------
//! Implements generic message-passing algorithms on factor graphs. See derived
//! classes for specific algorithms.

class drwnMessagePassingInference : public drwnInference
{
 public:
    static unsigned MAX_ITERATIONS; //!< maximum number of iterations

 protected:
    // forward and backward messages during each iteration
    vector<drwnTableFactor *> _forwardMessages;
    vector<drwnTableFactor *> _backwardMessages;
    vector<drwnTableFactor *> _oldForwardMessages;
    vector<drwnTableFactor *> _oldBackwardMessages;

    // computation tree: intermediate factors, and (atomic) factor operations
    vector<drwnTableFactor *> _intermediateFactors;
    vector<drwnFactorOperation *> _computations;

    // shared storage for intermediate factors
    vector<drwnTableFactorStorage *> _sharedStorage;

 public:
    drwnMessagePassingInference(const drwnFactorGraph& graph);
    //drwnMessagePassingInference(const drwnMessagePassingInference& inf);
    virtual ~drwnMessagePassingInference();

    void clear();
    bool inference();

 protected:
    virtual void initializeMessages();
    virtual void buildComputationGraph() = 0;
};

// drwnSumProductInference class --------------------------------------------
//! Implements sum-product inference.

class drwnSumProdInference : public drwnMessagePassingInference {
 public:
    drwnSumProdInference(const drwnFactorGraph& graph);
    ~drwnSumProdInference();

    void marginal(drwnTableFactor& belief) const;

 protected:
    void buildComputationGraph();
};

// drwnAsyncSumProductInference class ---------------------------------------
//! Implements asynchronous sum-product inference.

class drwnAsyncSumProdInference : public drwnSumProdInference {
 public:
    drwnAsyncSumProdInference(const drwnFactorGraph& graph);
    ~drwnAsyncSumProdInference();

 protected:
    void buildComputationGraph();
};

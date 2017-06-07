/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMapInference.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>
#include <set>
#include <map>
#include <list>

#include "drwnBase.h"
#include "drwnVarUniverse.h"
#include "drwnVarAssignment.h"
#include "drwnFactorGraph.h"
#include "drwnTableFactorOps.h"

// drwnMAPInference class --------------------------------------------------
//! Interface for various MAP inference (energy minimization) algorithms.
//!
//! Algorithms operate on a factor graph (see drwnFactorGraph) that is
//! assumed to be provided in energy function form, i.e.,
//! \f$E(x) = \sum_c \psi_c(x_c)\f$ where \f$P(x) \propto \exp\left\{-E(x)\right\}\f$
//! and the task is to find \f$\mathop{argmin}_x E(x)\f$.
//! A constant reference is maintained to the factor graph, so it is
//! important that the drwnFactorGraph object not be destoryed before
//! destroying the drwnMAPInference object.
//! \todo derive from drwnStdObjectIface to allow factory creation

class drwnMAPInference {
 protected:
    const drwnFactorGraph& _graph; //!< reference to initial clique potentials

 public:
    drwnMAPInference(const drwnFactorGraph& graph);
    drwnMAPInference(const drwnMAPInference& inf);
    virtual ~drwnMAPInference();

    //! Clear internally cached data (e.g., computation graph)
    virtual void clear() { /* do nothing */ };
    //! Run inference (or resume for iterative algorithms). Algorithms may
    //! initialize from \p mapAssignment if not empty. Returns an upper and
    //! lower bound (if available) of the minimum energy. The upper bound
    //! is the same as the energy of the best solution found (i.e., same as
    //! \p graph.getEnergy(mapAssignment)).
    virtual std::pair<double, double> inference(drwnFullAssignment& mapAssignment) = 0;
};

// drwnICMInference class --------------------------------------------------
//! Implements iterated conditional modes (ICM) MAP inference. This method
//! was first proposed in Besag, Royal Stats Society, 1986.

class drwnICMInference : public drwnMAPInference {
 public:
    drwnICMInference(const drwnFactorGraph& graph);
    ~drwnICMInference();

    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);
};

// drwnMessagePassingMAPInference class -------------------------------------
//! Implements generic message-passing algorithms on factor graphs. See derived
//! classes for specific algorithms.

class drwnMessagePassingMAPInference : public drwnMAPInference
{
 public:
    static unsigned MAX_ITERATIONS; //!< maximum number of iterations
    static double DAMPING_FACTOR;   //!< damping factor for updating messages

 protected:
    // forward and backward messages during each iteration
    std::vector<drwnTableFactor *> _forwardMessages;
    std::vector<drwnTableFactor *> _backwardMessages;
    std::vector<drwnTableFactor *> _oldForwardMessages;
    std::vector<drwnTableFactor *> _oldBackwardMessages;

    // computation tree: intermediate factors, and (atomic) factor operations
    std::vector<drwnTableFactor *> _intermediateFactors;
    std::vector<drwnFactorOperation *> _computations;

    // shared storage for intermediate factors
    std::vector<drwnTableFactorStorage *> _sharedStorage;

 public:
    drwnMessagePassingMAPInference(const drwnFactorGraph& graph);
    //drwnMessagePassingMAPInference(const drwnMessagePassingMAPInference& inf);
    virtual ~drwnMessagePassingMAPInference();

    void clear();
    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);

 protected:
    virtual void initializeMessages();
    virtual void buildComputationGraph() = 0;
    virtual void decodeBeliefs(drwnFullAssignment& mapAssignment);
};

// drwnMaxProductInference class --------------------------------------------
//! Implements max-product inference.
//!
//! \note Since the factor graph is expected to be in energy form
//! (i.e., negative log-probability), this algorithm is equivalent to
//! min-sum.

class drwnMaxProdInference : public drwnMessagePassingMAPInference {
 public:
    drwnMaxProdInference(const drwnFactorGraph& graph);
    ~drwnMaxProdInference();

 protected:
    void buildComputationGraph();
    void decodeBeliefs(drwnFullAssignment& mapAssignment);
};

// drwnAsyncMaxProductInference class ---------------------------------------
//! Implements asynchronous max-product (min-sum) inference.
//!
//! \note Since the factor graph is expected to be in energy form
//! (i.e., negative log-probability), this algorithm is equivalent to
//! min-sum.

class drwnAsyncMaxProdInference : public drwnMessagePassingMAPInference {
 public:
    drwnAsyncMaxProdInference(const drwnFactorGraph& graph);
    ~drwnAsyncMaxProdInference();

 protected:
    void buildComputationGraph();
    void decodeBeliefs(drwnFullAssignment& mapAssignment);
};

// drwnJunctionTreeInference class ------------------------------------------
//! Implements the junction tree algorithm for exact inference on a factor
//! graph using drwnAsyncMaxProdInference for the actual message passing.

class drwnJunctionTreeInference : public drwnMAPInference {
 public:
    drwnJunctionTreeInference(const drwnFactorGraph& graph);
    ~drwnJunctionTreeInference();

    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);
};

// drwnGEMPLPInference class ------------------------------------------------
//! Implements the generalized LP-based message passing algorithm of
//! Globerson and Jaakkola, NIPS 2007.

class drwnGEMPLPInference : public drwnMessagePassingMAPInference {
 protected:
    std::vector<drwnClique> _separators;      // list of all separators, S
    std::vector<drwnEdge> _edges;             // list of edges (c,s)
    std::vector<std::set<int> > _cliqueEdges; // mapping from c to _edges
    std::vector<std::set<int> > _separatorEdges; // mapping from s to _edges
    double _lastDualObjective;                // previous dual objecive value
    unsigned _maxIterations;                  // maximum number of iterations

 public:
    drwnGEMPLPInference(const drwnFactorGraph& graph);
    ~drwnGEMPLPInference();

    void clear();
    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);

 protected:
    void initializeMessages();
    void buildComputationGraph();
    void decodeBeliefs(drwnFullAssignment& mapAssignment);

    // helper functions for building clique-to-separator mapping
    int findSeparatorIndex(const drwnClique& cliqueA, const drwnClique& cliqueB);

    // computations for messages from clique \p cliqueId
    void addMessageUpdate(int cliqueId, const drwnClique& cliqueVars,
        const drwnTableFactor* psi = NULL);
};

// drwnSontag08Inference class ----------------------------------------------
//! Implements the incremental tightening of the LP MAP inference
//! algorithm from Sontag et al., UAI 2008.

class drwnSontag08Inference : public drwnGEMPLPInference {
 public:
    static unsigned WARMSTART_ITERATIONS; //!< maximum number of iterations
                                          //!< after adding clusters
    static unsigned MAX_CLIQUES_TO_ADD;   //!< number of cliques to add per cycle

 protected:
    std::vector<drwnClique> _additionalCliques;

 public:
    drwnSontag08Inference(const drwnFactorGraph& graph);
    ~drwnSontag08Inference();

    void clear();
    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);

 protected:
    void buildComputationGraph();

    //! Generates the set of clique candidates to test after each
    //! GEMPLP iteration. Derived classes can override this to implement
    //! different clique candidate strategies.
    virtual void findCliqueCandidates(std::map<drwnClique, std::vector<int> >& cliqueCandidateSet);
};

// drwnDualDecompositionInference class ------------------------------------
//! Implements dual decomposition MAP inference (see Komodakis and Paragios,
//! CVPR 2009 and works cited therein). Each factor is treated as a separate
//! slave.

class drwnDualDecompositionInference : public drwnMAPInference {
 public:
    static double INITIAL_ALPHA; //!< initial gradient step size
    static bool USE_MIN_MARGINALS; //!< use min-marginals for subgradients

 public:
    drwnDualDecompositionInference(const drwnFactorGraph& graph);
    ~drwnDualDecompositionInference();

    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);
};

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelSegCRFInference.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "Eigen/Core"

#include "cv.h"

#include "drwnBase.h"
#include "drwnPGM.h"
#include "drwnVision.h"

using namespace std;

// drwnPixelSegCRFInference class -----------------------------------------
//! Alpha-expansion inference for a pixel-level CRF model with unary,
//! contrast-dependent pairwise, and custom higher-order terms.
//!
//! \sa drwnSegImageInstance, drwnPixelSegModel
//! \sa \ref drwnProjMultiSeg
//!
//! \todo add custom addAuxiliaryTerms function

class drwnPixelSegCRFInference {
 public:
    //! default constructor
    drwnPixelSegCRFInference() { /* do nothing */ };
    //! destructor
    virtual ~drwnPixelSegCRFInference() { /* do nothing */ };

    //! Alpha-expansion inference given instance unary terms, pairwise strength and
    //! higher-order strength (modifies \p instance). When constructing graphs the S
    //! set (nomially 1) takes value \alpha and the T set (nominally 0) takes each
    //! variable's previous value.
    void alphaExpansion(drwnSegImageInstance *instance, double pairwiseStrength = 0.0,
        double higherOrderStrength = 0.0) const;

    //! Return the energy of a given instance labeling.
    double energy(const drwnSegImageInstance *instance, double pairwiseStrength = 0.0,
        double higherOrderStrength = 0.0) const;

 protected:
    //! adds node to the graph for any auxiliary variables (e.g., as required for
    //! higher-order terms)
    virtual void addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance *instance) const;

    //! add unary potentials to max-flow graph for a given alpha-expansion move
    //! and return constant energy component
    double addUnaryTerms(drwnMaxFlow *g, const drwnSegImageInstance *instance,
        int alpha, int varOffset = 0) const;

    //! add pairwise potentials to max-flow graph for a given alpha-expansion move
    //! and return constant energy component
    double addPairwiseTerms(drwnMaxFlow *g, double pairwiseStrength,
        const drwnSegImageInstance *instance, int alpha, int varOffset = 0) const;

    //! add higher-order potentials to max-flow graph for a given alpha-expansion move
    //! and return constant energy component
    virtual double addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
        const drwnSegImageInstance* instance, int alpha, int varOffset = 0) const;

    //! compute energy contribution from unary potentials
    double computeUnaryEnergy(const drwnSegImageInstance *instance) const;

    //! compute energy contribution from unary potentials
    double computePairwiseEnergy(const drwnSegImageInstance *instance, double pairwiseStrength) const;

    //! compute energy contribution from higher-order potentials
    virtual double computeHigherOrderEnergy(const drwnSegImageInstance *instance,
        double higherOrderStrength) const;
};

// drwnRobustPottsCRFInference -----------------------------------------------
//! Higher-order consistency potentials as robust P^N models,
//!   \sum_c \sum_k \psi^H_c(y_c; k)
//! where
//!   \psi^H_c(y_i; k) = \min \{1, \frac{1}{\eta N} \sum_i \ind{y_i \neq k} \}
//! and where 0.0 < \eta < 0.5
//!
//! Uses superpixels within the image instance to define cliques.

class drwnRobustPottsCRFInference : public drwnPixelSegCRFInference {
 public:
    double _eta;               //!< robustness parameter
    bool _bWeightBySize;       //!< weight potential by superpixel/clique size 

 public:
    drwnRobustPottsCRFInference(double eta = 0.25, bool bWeightBySize = true) : 
        _eta(eta), _bWeightBySize(bWeightBySize) { /* do nothing */ };
    ~drwnRobustPottsCRFInference() { /* do nothing */ };

 protected:
    void addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance* instance) const;

    double addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
        const drwnSegImageInstance* instance, int alpha, int varOffset = 0) const;

    double computeHigherOrderEnergy(const drwnSegImageInstance *instance,
        double higherOrderStrength) const;
};

// drwnWeightedRobustPottsCRFInference ---------------------------------------
//! Higher-order consistency potentials as robust P^N models with weighted
//! clique membership,
//!   \sum_c \sum_k \psi^H_c(y_c; k)
//! where
//!   \psi^H_c(y_i; k) = \min \{1, \frac{1}{\eta W_c} \sum_i \ind{y_i \neq k} w_i \}
//! and where 0.0 < \eta < 0.5, W_c = \sum_i w_i
//!
//! Uses auxiliaryData within the image instance to define cliques.

class drwnWeightedRobustPottsCRFInference : public drwnPixelSegCRFInference {
 public:
    double _eta;               //!< robustness parameter

 public:
    drwnWeightedRobustPottsCRFInference(double eta = 0.25) : _eta(eta) { /* do nothing */ };
    ~drwnWeightedRobustPottsCRFInference() { /* do nothing */ };

 protected:
    void addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance* instance) const;

    double addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
        const drwnSegImageInstance* instance, int alpha, int varOffset = 0) const;

    double computeHigherOrderEnergy(const drwnSegImageInstance *instance,
        double higherOrderStrength) const;
};

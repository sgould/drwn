/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGraphCutInference.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnFactorGraph.h"
#include "drwnMapInference.h"

// drwnAlphaExpansionInference -------------------------------------------------
//! Implements alpha-expansion inference using graph-cuts (see Boykov et al, 2001).
//! Factor graphs must be pairwise.

class drwnAlphaExpansionInference : public drwnMAPInference {
 public:
    drwnAlphaExpansionInference(const drwnFactorGraph& graph);
    ~drwnAlphaExpansionInference();

    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);
};

// drwnAlphaBetaSwapInference --------------------------------------------------
//! Implements alpha-beta swap inference using graph-cuts (see Boykov et al, 2001).
//! Factor graphs must be pairwise.

class drwnAlphaBetaSwapInference : public drwnMAPInference {
 public:
    drwnAlphaBetaSwapInference(const drwnFactorGraph& graph);
    ~drwnAlphaBetaSwapInference();

    std::pair<double, double> inference(drwnFullAssignment& mapAssignment);
};

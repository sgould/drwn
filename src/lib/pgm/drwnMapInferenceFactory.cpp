/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMAPInferenceFactory.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <vector>
#include <set>
#include <map>
#include <iterator>
#include <iomanip>

#include "drwnBase.h"
#include "drwnPGM.h"

// drwnMAPInferenceFactory -------------------------------------------------

drwnMAPInferenceFactory& drwnMAPInferenceFactory::get()
{
    static drwnMAPInferenceFactory factory;
    return factory;
}

list<string> drwnMAPInferenceFactory::getRegisteredClasses() const
{
    list<string> names;
    names.push_back(string("drwnICMInference"));
    names.push_back(string("drwnMaxProdInference"));
    names.push_back(string("drwnAsyncMaxProdInference"));
    names.push_back(string("drwnJunctionTreeInference"));
    names.push_back(string("drwnTRWSInference"));
    names.push_back(string("drwnGEMPLPInference"));
    names.push_back(string("drwnSontag08Inference"));
    names.push_back(string("drwnDualDecompositionInference"));
    names.push_back(string("drwnAlphaExpansionInference"));
    names.push_back(string("drwnAlphaBetaSwapInference"));
    names.push_back(string("drwnADLPInference"));
    return names;
}

drwnMAPInference *drwnMAPInferenceFactory::create(const char *name, const drwnFactorGraph& graph) const
{
    DRWN_ASSERT(name != NULL);

    if (string(name) == string("drwnICMInference")) {
        return new drwnICMInference(graph);
    } else if (string(name) == string("drwnMaxProdInference")) {
        return new drwnMaxProdInference(graph);
    } else if (string(name) == string("drwnAsyncMaxProdInference")) {
        return new drwnAsyncMaxProdInference(graph);
    } else if (string(name) == string("drwnJunctionTreeInference")) {
        return new drwnJunctionTreeInference(graph);
    } else if (string(name) == string("drwnTRWSInference")) {
        return new drwnTRWSInference(graph);
    } else if (string(name) == string("drwnGEMPLPInference")) {
        return new drwnGEMPLPInference(graph);
    } else if (string(name) == string("drwnSontag08Inference")) {
        return new drwnSontag08Inference(graph);
    } else if (string(name) == string("drwnDualDecompositionInference")) {
        return new drwnDualDecompositionInference(graph);
    } else if (string(name) == string("drwnAlphaExpansionInference")) {
        return new drwnAlphaExpansionInference(graph);
    } else if (string(name) == string("drwnAlphaBetaSwapInference")) {
        return new drwnAlphaBetaSwapInference(graph);
    } else if (string(name) == string("drwnADLPInference")) {
        return new drwnADLPInference(graph);
    }

    DRWN_LOG_ERROR("class \"" << name << "\" does not exist in drwnMAPInferenceFactory");
    return NULL;
}

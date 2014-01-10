/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnGraphCutInference.cpp
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

// drwnAlphaExpansionInference ---------------------------------------------

drwnAlphaExpansionInference::drwnAlphaExpansionInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnAlphaExpansionInference::~drwnAlphaExpansionInference()
{
    // do nothing
}

double drwnAlphaExpansionInference::inference(drwnFullAssignment& mapAssignment)
{
    // check factor graph is pairwise
    for (int i = 0; i < _graph.numFactors(); i++) {
        DRWN_ASSERT(_graph[i]->size() <= 2);
    }

    // initialize MAP assignment
    const drwnVarUniversePtr pUniverse(_graph.getUniverse());
    const int nVariables = pUniverse->numVariables();
    mapAssignment.resize(nVariables, 0);
    double bestEnergy = _graph.getEnergy(mapAssignment);
    const int maxAlpha = pUniverse->maxCardinality();
    if (maxAlpha == 1) return bestEnergy;

    drwnFullAssignment assignment(mapAssignment);

    drwnMaxFlow *g = new drwnBKMaxFlow(nVariables);
    g->addNodes(nVariables);

    // iterate until convergence
    bool bChanged = true;
    int lastChanged = -1;
    int nonSubmodularMoves = 0;
    for (int nCycle = 0; bChanged; nCycle += 1) {
        bChanged = false;
        for (int alpha = 0; alpha < maxAlpha; alpha++) {
            if (alpha == lastChanged)
                break;

            // construct graph
            bool bNonSubmodularMove = false;
            g->reset();
            for (int i = 0; i < _graph.numFactors(); i++) {
                const drwnTableFactor *phi = _graph[i];
                if (phi->size() == 1) {
                    // unary
                    const int u = phi->varId(0);
                    const double A = (*phi)[mapAssignment[u]];
                    const double B = (*phi)[alpha % pUniverse->varCardinality(u)];
                    g->addSourceEdge(u, A);
                    g->addTargetEdge(u, B);
                } else if (phi->size() == 2) {
                    // pairwise
                    const int u = phi->varId(0);
                    const int v = phi->varId(1);

                    const int uAlpha = alpha % pUniverse->varCardinality(u);
                    const int vAlpha = alpha % pUniverse->varCardinality(v);

                    if ((uAlpha == mapAssignment[u]) && (vAlpha == mapAssignment[v]))
                        continue;

                    double A = (*phi)[phi->indexOf(v, mapAssignment[v], phi->indexOf(u, mapAssignment[u]))];
                    double B = (*phi)[phi->indexOf(v, vAlpha, phi->indexOf(u, mapAssignment[u]))];
                    double C = (*phi)[phi->indexOf(v, mapAssignment[v], phi->indexOf(u, uAlpha))];
                    double D = (*phi)[phi->indexOf(v, vAlpha, phi->indexOf(u, uAlpha))];

                    if (uAlpha == mapAssignment[u]) {
                        g->addSourceEdge(v, A);
                        g->addTargetEdge(v, B);
                    } else if (vAlpha == mapAssignment[v]) {
                        g->addSourceEdge(u, A);
                        g->addTargetEdge(u, C);
                    } else {
                        // check for submodularity
                        if (A + D > C + B) {
                            // truncate non-submodular functions
                            bNonSubmodularMove = true;
                            const double delta = A + D - C - B;
                            A -= delta / 3 - DRWN_EPSILON;
                            C += delta / 3 + DRWN_EPSILON;
                            B = A + D - C + DRWN_EPSILON;
                        }

                        g->addSourceEdge(u, A);
                        g->addTargetEdge(u, D);

                        B -= A; C -= D; B += DRWN_EPSILON; C += DRWN_EPSILON;
                        DRWN_ASSERT_MSG(B + C >= 0, "B = " << B << ", C = " << C);
                        if (B < 0) {
                            g->addTargetEdge(v, B);
                            g->addTargetEdge(u, -B);
                            g->addEdge(v, u, 0.0, B + C);
                        } else if (C < 0) {
                            g->addTargetEdge(v, -C);
                            g->addTargetEdge(u, C);
                            g->addEdge(v, u, B + C, 0.0);
                        } else {
                            g->addEdge(v, u, B, C);
                        }
                    }
                }
            }

            // run inference
            if (bNonSubmodularMove) {
                nonSubmodularMoves += 1;
            }

            g->solve();
            for (int i = 0; i < nVariables; i++) {
                assignment[i] = (g->inSetS(i) ? alpha % pUniverse->varCardinality(i) : mapAssignment[i]);
            }
            double e = _graph.getEnergy(assignment);

            DRWN_LOG_DEBUG("...cycle " << nCycle << ", iteration " << alpha << " has energy " << e);
            if (e < bestEnergy) {
                bestEnergy = e;
                mapAssignment = assignment;
                lastChanged = alpha;
                bChanged = true;
            }
        }
    }

    // free graph
    delete g;

    DRWN_LOG_DEBUG("..." << nonSubmodularMoves << " non-submodular moves");
    return bestEnergy;
}

// drwnAlphaBetaSwapInference ----------------------------------------------

drwnAlphaBetaSwapInference::drwnAlphaBetaSwapInference(const drwnFactorGraph& graph) :
    drwnMAPInference(graph)
{
    // do nothing
}

drwnAlphaBetaSwapInference::~drwnAlphaBetaSwapInference()
{
    // do nothing
}

double drwnAlphaBetaSwapInference::inference(drwnFullAssignment& mapAssignment)
{
    // check factor graph is pairwise
    for (int i = 0; i < _graph.numFactors(); i++) {
        DRWN_ASSERT(_graph[i]->size() <= 2);
    }

    // initialize MAP assignment
    const drwnVarUniversePtr pUniverse(_graph.getUniverse());
    const int nVariables = pUniverse->numVariables();
    mapAssignment.resize(nVariables, 0);
    double bestEnergy = _graph.getEnergy(mapAssignment);
    const int maxLabel = pUniverse->maxCardinality();
    if (maxLabel == 1) return bestEnergy;

    drwnFullAssignment assignment(mapAssignment);

    drwnMaxFlow *g = new drwnBKMaxFlow(nVariables);
    g->addNodes(nVariables);

    // iterate until convergence
    bool bChanged = true;
    pair<int, int> lastChanged = make_pair(-1, -1);
    int nonSubmodularMoves = 0;
    for (int nCycle = 0; bChanged; nCycle += 1) {
        bChanged = false;
        for (int alpha = 0; alpha < maxLabel - 1; alpha++) {
            for (int beta = alpha + 1; beta < maxLabel; beta++) {
                if (lastChanged == make_pair(alpha, beta))
                    break;

                // construct graph
                bool bNonSubmodularMove = false;
                g->reset();
                for (int i = 0; i < _graph.numFactors(); i++) {
                    const drwnTableFactor *phi = _graph[i];
                    if (phi->size() == 1) {
                        // unary
                        const int u = phi->varId(0);
                        if ((mapAssignment[u] == alpha % pUniverse->varCardinality(u)) ||
                            (mapAssignment[u] == beta % pUniverse->varCardinality(u))) {
                            const double A = (*phi)[beta % pUniverse->varCardinality(u)];
                            const double B = (*phi)[alpha % pUniverse->varCardinality(u)];
                            g->addSourceEdge(u, A);
                            g->addTargetEdge(u, B);
                        }
                    } else if (phi->size() == 2) {
                        // pairwise
                        const int u = phi->varId(0);
                        const int v = phi->varId(1);

                        const int uAlpha = alpha % pUniverse->varCardinality(u);
                        const int vAlpha = alpha % pUniverse->varCardinality(v);
                        const int uBeta = beta % pUniverse->varCardinality(u);
                        const int vBeta = beta % pUniverse->varCardinality(v);

                        const bool uNotAlphaOrBeta = (mapAssignment[u] != uAlpha) && (mapAssignment[u] != uBeta);
                        const bool vNotAlphaOrBeta = (mapAssignment[v] != vAlpha) && (mapAssignment[v] != vBeta);
                        if (uNotAlphaOrBeta && vNotAlphaOrBeta)
                            continue;

                        if (uNotAlphaOrBeta) {
                            const int uIndx = phi->indexOf(u, mapAssignment[u]);
                            const double A = (*phi)[phi->indexOf(v, vBeta, uIndx)];
                            const double B = (*phi)[phi->indexOf(v, vAlpha, uIndx)];
                            g->addSourceEdge(v, A);
                            g->addTargetEdge(v, B);

                        } else if (vNotAlphaOrBeta) {
                            const int vIndx = phi->indexOf(v, mapAssignment[v]);
                            const double A = (*phi)[phi->indexOf(u, uBeta, vIndx)];
                            const double B = (*phi)[phi->indexOf(u, uAlpha, vIndx)];
                            g->addSourceEdge(u, A);
                            g->addTargetEdge(u, B);

                        } else {
                            double A = (*phi)[phi->indexOf(v, vBeta, phi->indexOf(u, uBeta))];
                            double B = (*phi)[phi->indexOf(v, vAlpha, phi->indexOf(u, uBeta))];
                            double C = (*phi)[phi->indexOf(v, vBeta, phi->indexOf(u, uAlpha))];
                            double D = (*phi)[phi->indexOf(v, vAlpha, phi->indexOf(u, uAlpha))];

                            // check for submodularity
                            if (A + D > C + B) {
                                // truncate non-submodular functions
                                bNonSubmodularMove = true;
                                const double delta = A + D - C - B;
                                A -= delta / 3 - DRWN_EPSILON;
                                C += delta / 3 + DRWN_EPSILON;
                                B = A + D - C + DRWN_EPSILON;
                            }

                            g->addSourceEdge(u, A);
                            g->addTargetEdge(u, D);

                            B -= A; C -= D; B += DRWN_EPSILON; C += DRWN_EPSILON;
                            DRWN_ASSERT_MSG(B + C >= 0, "B = " << B << ", C = " << C);
                            if (B < 0) {
                                g->addTargetEdge(v, B);
                                g->addTargetEdge(u, -B);
                                g->addEdge(v, u, 0.0, B + C);
                            } else if (C < 0) {
                                g->addTargetEdge(v, -C);
                                g->addTargetEdge(u, C);
                                g->addEdge(v, u, B + C, 0.0);
                            } else {
                                g->addEdge(v, u, B, C);
                            }
                        }
                    }
                }

                // run inference
                if (bNonSubmodularMove) {
                    nonSubmodularMoves += 1;
                }

                g->solve();
                for (int i = 0; i < nVariables; i++) {
                    const int iAlpha = alpha % pUniverse->varCardinality(i);
                    const int iBeta = beta % pUniverse->varCardinality(i);
                    if ((mapAssignment[i] == iAlpha) || (mapAssignment[i] == iBeta)) {
                        assignment[i] = (g->inSetS(i) ? iAlpha : iBeta);
                    }
                }
                double e = _graph.getEnergy(assignment);
                
                DRWN_LOG_DEBUG("...cycle " << nCycle << ", iteration (" << alpha << ", "
                    << beta << ") has energy " << e);
                if (e < bestEnergy) {
                    bestEnergy = e;
                    mapAssignment = assignment;
                    lastChanged = make_pair(alpha, beta);
                    bChanged = true;
                }
            }
        }
    }

    // free graph
    delete g;

    DRWN_LOG_DEBUG("..." << nonSubmodularMoves << " non-submodular moves");
    return bestEnergy;
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnPixelSegCRFInference.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

#include "drwnPixelSegCRFInference.h"

using namespace std;

// drwnPixelSegCRFInference class -----------------------------------------

void drwnPixelSegCRFInference::alphaExpansion(drwnSegImageInstance *instance,
    double pairwiseStrength, double higherOrderStrength) const
{
    DRWN_ASSERT(instance != NULL);
    DRWN_ASSERT(instance->unaries.size() == (unsigned)instance->size());
    DRWN_ASSERT(pairwiseStrength >= 0.0);

    // check cardinality
    const int nVariables = instance->size();
    const int maxAlpha = (int)instance->unaries[0].size();
    for (int i = 1; i < nVariables; i++) {
        DRWN_ASSERT(instance->unaries[i].size() == (unsigned)maxAlpha);
    }

    // initialize using unary
    int varIndx = 0;
    double eUnary = 0.0;
    for (int y = 0; y < instance->height(); y++) {
        for (int x = 0; x < instance->width(); x++, varIndx++) {
            instance->pixelLabels(y, x) = drwn::argmin(instance->unaries[varIndx]);
            eUnary += instance->unaries[varIndx][instance->pixelLabels(y, x)];
        }
    }
    DRWN_LOG_DEBUG("...unary energy is " << eUnary);

    // return if no pairwise or higher-order terms
    if ((pairwiseStrength == 0.0) && (higherOrderStrength == 0.0)) {
        return;
    }

    int hConstructGraph = drwnCodeProfiler::getHandle("alphaExpansion.constructGraph");
    int hMinCutIteration = drwnCodeProfiler::getHandle("alphaExpansion.mincut");

    drwnMaxFlow *g = new drwnBKMaxFlow(nVariables);
    g->addNodes(nVariables);
    addAuxiliaryVariables(g, instance);

    bool bChanged = true;
    int lastChanged = -1;
    double minEnergy = numeric_limits<double>::max();
    for (int nCycle = 0; bChanged; nCycle += 1) {
        bChanged = false;
        for (int alpha = 0; alpha < maxAlpha; alpha++) {
            if (alpha == lastChanged)
                break;

            drwnCodeProfiler::tic(hConstructGraph);
            g->reset();
            double e = 0.0;

            // add unary terms
            e += addUnaryTerms(g, instance, alpha);

            // add pairwise terms
            e += addPairwiseTerms(g, pairwiseStrength, instance, alpha);

            // add higher order terms
            e += addHigherOrderTerms(g, higherOrderStrength, instance, alpha);

            // run inference
            drwnCodeProfiler::toc(hConstructGraph);
            drwnCodeProfiler::tic(hMinCutIteration);
            e += g->solve();

            DRWN_LOG_DEBUG("...cycle " << nCycle << ", iteration " << alpha
                << " has energy " << e << " (min. " << minEnergy << ")");
            int freeVarsCount = 0;
            if (e < minEnergy) {
                minEnergy = e;
                lastChanged = alpha;
                bChanged = true;

                varIndx = 0;
                for (int y = 0; y < instance->height(); y++) {
                    for (int x = 0; x < instance->width(); x++, varIndx++) {
                        if (g->inSetS(varIndx)) {
                            instance->pixelLabels(y, x) = alpha;
                        }

                        if (!g->inSetS(varIndx) && !g->inSetT(varIndx)) {
                            freeVarsCount += 1;
                            instance->pixelLabels(y, x) = alpha;
                        }
                    }
                }
            }
            DRWN_LOG_DEBUG("..." << freeVarsCount << " variables can be labeled arbitrarily");

            drwnCodeProfiler::toc(hMinCutIteration);
        }
    }

    // free graph
    delete g;
}

double drwnPixelSegCRFInference::energy(const drwnSegImageInstance *instance, double pairwiseStrength,
    double higherOrderStrength) const
{
    DRWN_ASSERT(instance != NULL);
    double e = 0.0;

    // add energy from unary terms
    e += computeUnaryEnergy(instance);

    // add energy from pairwise terms
    if (pairwiseStrength != 0.0) {
        e += computePairwiseEnergy(instance, pairwiseStrength);
    }

    // add energy from higher order terms
    if (higherOrderStrength != 0.0) {
        e += computeHigherOrderEnergy(instance, higherOrderStrength);
    }

    return e;
}

void drwnPixelSegCRFInference::addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance *instance) const
{
    // do nothing
}

// alpha-expansion graph construction
double drwnPixelSegCRFInference::addUnaryTerms(drwnMaxFlow *g, const drwnSegImageInstance *instance,
    int alpha, int varOffset) const
{
    int varIndx = 0;
    for (int y = 0; y < instance->height(); y++) {
        for (int x = 0; x < instance->width(); x++, varIndx++) {
            if (instance->pixelLabels(y, x) != alpha) {
                g->addSourceEdge(varIndx + varOffset, instance->unaries[varIndx][instance->pixelLabels(y, x)]);
                g->addTargetEdge(varIndx + varOffset, instance->unaries[varIndx][alpha]);
            } else {
                g->addConstant(instance->unaries[varIndx][alpha]);
            }
        }
    }

    return 0.0;
}

double drwnPixelSegCRFInference::addPairwiseTerms(drwnMaxFlow *g, double pairwiseStrength,
    const drwnSegImageInstance *instance, int alpha, int varOffset) const
{
    // return immediately if pairwise strength is zero
    if (pairwiseStrength == 0.0) return 0.0;
    DRWN_ASSERT(pairwiseStrength > 0.0);

    // add horizontal pairwise terms
    for (int y = 0; y < instance->height(); y++) {
        for (int x = 1; x < instance->width(); x++) {
            const int labelA = instance->pixelLabels(y, x);
            const int labelB = instance->pixelLabels(y, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            const int u = instance->pixel2Indx(x, y) + varOffset;
            const int v = instance->pixel2Indx(x - 1, y) + varOffset;

            const double w = pairwiseStrength * instance->contrast.contrastW(x, y);

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    // add vertical pairwise terms
    for (int y = 1; y < instance->height(); y++) {
        for (int x = 0; x < instance->width(); x++) {
            const int labelA = instance->pixelLabels(y, x);
            const int labelB = instance->pixelLabels(y - 1, x);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            const int u = instance->pixel2Indx(x, y) + varOffset;
            const int v = instance->pixel2Indx(x, y - 1) + varOffset;

            const double w = pairwiseStrength * instance->contrast.contrastN(x, y);

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    // add diagonal pairwise terms
    for (int y = 1; y < instance->height(); y++) {
        for (int x = 1; x < instance->width(); x++) {
            const int labelA = instance->pixelLabels(y, x);
            const int labelB = instance->pixelLabels(y - 1, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            const int u = instance->pixel2Indx(x, y) + varOffset;
            const int v = instance->pixel2Indx(x - 1, y - 1) + varOffset;

            const double w = pairwiseStrength * instance->contrast.contrastNW(x, y);

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    for (int y = 1; y < instance->height(); y++) {
        for (int x = 1; x < instance->width(); x++) {
            const int labelA = instance->pixelLabels(y - 1, x);
            const int labelB = instance->pixelLabels(y, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            const int u = instance->pixel2Indx(x, y - 1) + varOffset;
            const int v = instance->pixel2Indx(x - 1, y) + varOffset;

            const double w = pairwiseStrength * instance->contrast.contrastSW(x, y - 1);

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    return 0.0;
}

double drwnPixelSegCRFInference::addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
    const drwnSegImageInstance *instance, int alpha, int varOffset) const
{
    return 0.0;
}

double drwnPixelSegCRFInference::computeUnaryEnergy(const drwnSegImageInstance *instance) const
{
    double e = 0.0;

    int indx = 0;
    for (int y = 0; y < instance->height(); y++) {
        for (int x = 0; x < instance->width(); x++, indx++) {
            e += instance->unaries[indx][instance->pixelLabels(y, x)];
        }
    }

    return e;
}

double drwnPixelSegCRFInference::computePairwiseEnergy(const drwnSegImageInstance *instance, double pairwiseStrength) const
{
    double e = 0.0;

    // horizontal pairwise
    for (int y = 0; y < instance->height(); y++) {
        for (int x = 1; x < instance->width(); x++) {
            if (instance->pixelLabels(y, x) != instance->pixelLabels(y, x - 1)) {
                e +=  instance->contrast.contrastW(x, y);
            }
        }
    }

    // vertical pairwise
    for (int y = 1; y < instance->height(); y++) {
        for (int x = 0; x < instance->width(); x++) {
            if (instance->pixelLabels(y, x) != instance->pixelLabels(y - 1, x)) {
                e += instance->contrast.contrastN(x, y);
            }
        }
    }

    // diagonal pairwise
    for (int y = 1; y < instance->height(); y++) {
        for (int x = 1; x < instance->width(); x++) {
            if (instance->pixelLabels(y, x) != instance->pixelLabels(y - 1, x - 1)) {
                e += instance->contrast.contrastNW(x, y);
            }

            if (instance->pixelLabels(y - 1, x) != instance->pixelLabels(y, x - 1)) {
                e += instance->contrast.contrastSW(x, y - 1);
            }
        }
    }

    return pairwiseStrength * e;
}

double drwnPixelSegCRFInference::computeHigherOrderEnergy(const drwnSegImageInstance *instance,
    double higherOrderStrength) const
{
    return 0.0;
}

// drwnRobustPottsCRFInference -----------------------------------------------

void drwnRobustPottsCRFInference::addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance* instance) const
{
    const int nLabels = (int)instance->unaries[0].size();
    g->addNodes(instance->superpixels.size() * nLabels);
}

double drwnRobustPottsCRFInference::addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
    const drwnSegImageInstance* instance, int alpha, int varOffset) const
{
    // return immediately if high order weight is zero
    if ((higherOrderStrength == 0.0) || instance->superpixels.empty()) return 0.0;
    DRWN_ASSERT(higherOrderStrength > 0.0);

    // determine number of labels
    const int nLabels = (int)instance->unaries[0].size();

    // add auxiliary variables, one for each (superpixel, label) pair
    int auxVarIndx = instance->height() * instance->width();
    DRWN_ASSERT((int)g->numNodes() != auxVarIndx);

    // add a higher order term for each superpixel
    cv::Mat segMask(instance->height(), instance->width(), CV_8UC1);
    for (unsigned segIndx = 0; segIndx < (unsigned)instance->superpixels.size(); segIndx++) {

        const int segWeight = instance->superpixels.pixels(segIndx);
        if (segWeight == 0) continue;
        instance->superpixels.mask(segIndx, segMask);

        const double segStrength = _bWeightBySize ? higherOrderStrength * (double)segWeight : higherOrderStrength;
        const double scalingFactor = segStrength / (_eta * (double)segWeight);

        // add term for label equal to alpha
        int varIndx = varOffset;
        for (int y = 0; y < instance->height(); y++) {
            const unsigned char *w = segMask.ptr<const unsigned char>(y);
            for (int x = 0; x < instance->width(); x++, varIndx++) {
                if ((w[x] != 0x00) && (instance->pixelLabels(y, x) != alpha)) {
                    g->addEdge(auxVarIndx, varIndx, scalingFactor);
                }
            }
        }
        g->addSourceEdge(auxVarIndx, segStrength);

        // increment auxiliary variable
        auxVarIndx += 1;

        // add terms for labels not equal to alpha
        for (int lblIndx = 0; lblIndx < nLabels; lblIndx++) {
            if (lblIndx == alpha) continue;

            varIndx = varOffset;
            int lblWeight = 0;
            for (int y = 0; y < instance->height(); y++) {
                const unsigned char *w = segMask.ptr<const unsigned char>(y);
                for (int x = 0; x < instance->width(); x++, varIndx++) {
                    if ((w[x] != 0x00) && (instance->pixelLabels(y, x) == lblIndx)) {
                        lblWeight += 1;
                        g->addEdge(varIndx, auxVarIndx, scalingFactor);
                    }
                }
            }
            g->addSourceEdge(auxVarIndx, (segWeight - lblWeight) * scalingFactor - segStrength);
            g->addConstant(segStrength);

            // increment auxiliary variable
            auxVarIndx += 1;
        }
    }

    return 0.0;
}

double drwnRobustPottsCRFInference::computeHigherOrderEnergy(const drwnSegImageInstance *instance,
    double higherOrderStrength) const
{
    // determine number of labels
    const int nLabels = (int)instance->unaries[0].size();

    cv::Mat segMask(instance->height(), instance->width(), CV_8UC1);
    vector<int> labelDistribution(nLabels);

    // add energy for each superpixel
    double e = 0.0;
    for (unsigned segIndx = 0; segIndx < (unsigned)instance->superpixels.size(); segIndx++) {

        const int segWeight = instance->superpixels.pixels(segIndx);
        if (segWeight == 0) continue;

        instance->superpixels.mask(segIndx, segMask);
        std::fill(labelDistribution.begin(), labelDistribution.end(), 0);

        // count labels
        for (int y = 0; y < instance->height(); y++) {
            const unsigned char *w = segMask.ptr<const unsigned char>(y);
            for (int x = 0; x < instance->width(); x++) {
                if (w[x] != 0x00) {
                    labelDistribution[instance->pixelLabels(y, x)] += 1;
                }
            }
        }

        // find number of violations
        const int violations = segWeight - drwn::maxElem(labelDistribution);

        double penalty;
        if (_bWeightBySize) {
            penalty = std::min((double)violations / _eta, (double)segWeight);
        } else {
            penalty = std::min((double)violations / (_eta * segWeight), 1.0);
        }

        e += penalty;
    }

    return higherOrderStrength * e;
}

// drwnWeightedRobustPottsCRFInference ---------------------------------------

void drwnWeightedRobustPottsCRFInference::addAuxiliaryVariables(drwnMaxFlow *g, const drwnSegImageInstance* instance) const
{
    const int nLabels = (int)instance->unaries[0].size();
    g->addNodes(instance->auxiliaryData.size() * nLabels);
}

double drwnWeightedRobustPottsCRFInference::addHigherOrderTerms(drwnMaxFlow *g, double higherOrderStrength,
    const drwnSegImageInstance* instance, int alpha, int varOffset) const
{
    // return immediately if high order weight is zero
    if ((higherOrderStrength == 0.0) || instance->auxiliaryData.empty()) return 0.0;
    DRWN_ASSERT(higherOrderStrength > 0.0);

    // determine number of labels
    const int nLabels = (int)instance->unaries[0].size();

    // add auxiliary variables, one for each (soft segmentation, label) pair
    int auxVarIndx = instance->height() * instance->width();
    DRWN_ASSERT((int)g->numNodes() != auxVarIndx);

    // add a higher order term for each soft segmentation
    for (unsigned segIndx = 0; segIndx < instance->auxiliaryData.size(); segIndx++) {

        float segWeight = 0.0f;
        cv::MatConstIterator_<float> it = instance->auxiliaryData[segIndx].begin<float>();
        cv::MatConstIterator_<float> it_end = instance->auxiliaryData[segIndx].end<float>();
        for ( ; it != it_end; ++it) {
            segWeight += *it;
        }
        if (segWeight == 0.0f) continue;

        //const double segStrength = higherOrderStrength;
        const double segStrength = higherOrderStrength * (double)segWeight;
        const double scalingFactor = segStrength / (_eta * (double)segWeight);

        // add term for label equal to alpha
        int varIndx = varOffset;
        for (int y = 0; y < instance->height(); y++) {
            const float *w = instance->auxiliaryData[segIndx].ptr<const float>(y);
            for (int x = 0; x < instance->width(); x++, varIndx++) {
                if (instance->pixelLabels(y, x) != alpha) {
                    g->addEdge(auxVarIndx, varIndx, w[x] * scalingFactor);
                }
            }
        }
        g->addSourceEdge(auxVarIndx, segStrength);

        // increment auxiliary variable
        auxVarIndx += 1;

        // add terms for labels not equal to alpha
        for (int lblIndx = 0; lblIndx < nLabels; lblIndx++) {
            if (lblIndx == alpha) continue;

            varIndx = varOffset;
            float lblWeight = 0.0f;
            for (int y = 0; y < instance->height(); y++) {
                const float *w = instance->auxiliaryData[segIndx].ptr<const float>(y);
                for (int x = 0; x < instance->width(); x++, varIndx++) {
                    if (instance->pixelLabels(y, x) == lblIndx) {
                        lblWeight += w[x];
                        g->addEdge(varIndx, auxVarIndx, w[x] * scalingFactor);
                    }
                }
            }
            g->addSourceEdge(auxVarIndx, (segWeight - lblWeight) * scalingFactor - segStrength);
            g->addConstant(segStrength);

            // increment auxiliary variable
            auxVarIndx += 1;
        }
    }

    return 0.0;
}

double drwnWeightedRobustPottsCRFInference::computeHigherOrderEnergy(const drwnSegImageInstance *instance,
    double higherOrderStrength) const
{
    DRWN_TODO;
    return 0.0;
}

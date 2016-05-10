/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexLoadPatchMatchGraph.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// eigen matrix library headers
#include "Eigen/Core"

// matlab headers
#include "mex.h"
#include "matrix.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

// main -----------------------------------------------------------------------

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: graph = mexLoadPatchMatchGraph(filebase, [options]);\n");
    mexPrintf("  filebase :: base filename for the PatchMatchGraph\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if ((nrhs != 1) && (nrhs != 2)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 2) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    char *filebase = mxArrayToString(prhs[0]);

    // load PatchMatchGraph
    drwnPatchMatchGraph graph;
    DRWN_LOG_MESSAGE("Loading PatchMatchGraph from " << filebase << "...");
    graph.read(filebase);
    DRWN_LOG_MESSAGE("...graph has " << graph.size() << " images");

    mxFree(filebase);

    // convert to Matlab data structure
    if (nlhs == 1) {
        const char *fnames[] = {"dir", "ext", "patchWidth", "patchHeight", "imageNames", "imageSizes", "matches"};
        plhs[0] = mxCreateStructMatrix(1, 1, 7, fnames);

        // meta-data
        mxSetFieldByNumber(plhs[0], 0, 0, mxCreateString(graph.imageDirectory.c_str()));
        mxSetFieldByNumber(plhs[0], 0, 1, mxCreateString(graph.imageExtension.c_str()));
        mxSetFieldByNumber(plhs[0], 0, 2, mxCreateDoubleScalar((double)graph.patchWidth()));
        mxSetFieldByNumber(plhs[0], 0, 3, mxCreateDoubleScalar((double)graph.patchHeight()));

        // image names
        mxArray *imageNames = mxCreateCellMatrix(graph.size(), 1);
        for (unsigned i = 0; i < graph.size(); i++) {
            mxSetCell(imageNames, i, mxCreateString(graph[i].name().c_str()));
        }
        mxSetFieldByNumber(plhs[0], 0, 4, imageNames);

        // image sizes
        mxArray *imageSizes = mxCreateDoubleMatrix(graph.size(), 2, mxREAL);
        double *px = mxGetPr(imageSizes);
        for (unsigned i = 0; i < graph.size(); i++) {
            px[i] = (double)graph[i][0].width();
            px[i + graph.size()] = (double)graph[i][0].height();
        }
        mxSetFieldByNumber(plhs[0], 0, 5, imageSizes);        

        // matches
        size_t numMatches = 0;
        for (unsigned i = 0; i < graph.size(); i++) {
            for (unsigned j = 0; j < graph[i].levels(); j++) {
                for (unsigned k = 0; k < graph[i][j].size(); k++) {
                    numMatches += graph[i][j][k].size();
                }
            }
        }

        mxArray *matches = mxCreateDoubleMatrix(numMatches, 11, mxREAL);
        px = mxGetPr(matches);
        for (unsigned i = 0; i < graph.size(); i++) {
            for (unsigned j = 0; j < graph[i].levels(); j++) {
                for (unsigned k = 0; k < graph[i][j].size(); k++) {
                    const drwnPatchMatchEdgeList& e = graph[i][j][k];
                    for (drwnPatchMatchEdgeList::const_iterator it = e.begin(); it != e.end(); ++it) {
                        px[0] = (double)(i + 1); // src_img

                        cv::Point tl = graph[i][j].index2pixel(k);
                        cv::Point br = cv::Point(graph.patchWidth(), graph.patchHeight());
                        tl = graph[i].mapPixel(tl, j, 0);
                        br = graph[i].mapPixel(br, j, 0);
                        px[1 * numMatches] = (double)tl.x + 1; // src_patch
                        px[2 * numMatches] = (double)tl.y + 1;
                        px[3 * numMatches] = (double)br.x;
                        px[4 * numMatches] = (double)br.y;

                        px[5 * numMatches] = (double)(it->targetNode.imgIndx + 1); // dst_img

                        tl = cv::Point(it->targetNode.xPosition, it->targetNode.yPosition);
                        br = cv::Point(graph.patchWidth(), graph.patchHeight());
                        tl = graph[it->targetNode.imgIndx].mapPixel(tl, it->targetNode.imgScale, 0);
                        br = graph[it->targetNode.imgIndx].mapPixel(br, it->targetNode.imgScale, 0);
                        px[6 * numMatches] = (double)tl.x + 1; // dst_patch
                        px[7 * numMatches] = (double)tl.y + 1;
                        px[8 * numMatches] = (double)br.x;
                        px[9 * numMatches] = (double)br.y;

                        px[10 * numMatches] = (double)it->matchScore; // score

                        px++;
                    }
                }
            }
        }

        // src_img, src_patch (4), dst_img, dst_patch (4), score
        mxSetFieldByNumber(plhs[0], 0, 6, matches);
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

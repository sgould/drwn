/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexAnalyseClassifier.cpp
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
#include "drwnML.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: [confusion, prcurves] = mexAnalyseClassifier(scores, labels, [options]);\n");
    mexPrintf("  scores   :: N-by-K matrix of classifier scores\n");
    mexPrintf("  labels   :: N-vector of labels (0-based)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if (nrhs == 0) {
        usage();
        return;
    }

    if ((nrhs < 2) || (nrhs > 3)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    DRWN_ASSERT_MSG(mxGetM(prhs[1]) == mxGetM(prhs[0]), "mismatch between scores and labels");
    if (mxIsEmpty(prhs[0])) return;

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 3) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse scores
    vector<vector<double> > scores;
    drwnMatlabUtils::mxArrayToVector(prhs[0], scores);    
    vector<int> predictions = drwn::argmaxs(scores);

    // parse labels
    vector<int> labels(scores.size());
    int maxLabel = 0;
    switch (mxGetClassID(prhs[1])) {
    case mxDOUBLE_CLASS:
        {
            const double *p = mxGetPr(prhs[1]);
            for (unsigned i = 0; i < labels.size(); i++) {
                labels[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    case mxINT32_CLASS:
        {
            const int32_t *p = (const int32_T *)mxGetData(prhs[1]);
            for (unsigned i = 0; i < labels.size(); i++) {
                labels[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    case mxLOGICAL_CLASS:
        {
            const mxLogical *p = (const mxLogical *)mxGetData(prhs[1]);
            for (unsigned i = 0; i < labels.size(); i++) {
                labels[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized datatype, try labels = double(labels);");
    }

    // compute confusion matrix
    drwnConfusionMatrix confusion(maxLabel + 1, scores[0].size());
    confusion.accumulate(labels, predictions);
    if (drwnLogger::checkLogLevel(DRWN_LL_VERBOSE)) {
        confusion.printCounts(cout);
    }

    // output confusion
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleMatrix(confusion.numRows(), confusion.numCols(), mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (int j = 0; j < confusion.numCols(); j++) {
            for (int i = 0; i < confusion.numRows(); i++) {
                *p++ = confusion(i, j);
            }
        }
    }

    // output pr curves
    if (nlhs >= 2) {
        if (maxLabel == 1) {
            drwnPRCurve curve;
            for (unsigned i = 0; i < labels.size(); i++) {
                if (labels[i] == 1) {
                    curve.accumulatePositives(scores[i][1] - scores[i][0]);
                } else if (labels[i] == 0) {
                    curve.accumulateNegatives(scores[i][1] - scores[i][0]);
                }
            }
            
            vector<pair<double, double> > p = curve.getCurve();
            plhs[1] = mxCreateDoubleMatrix(p.size(), 2, mxREAL);
            double *q = mxGetPr(plhs[1]);
            for (unsigned i = 0; i < p.size(); i++) {
                q[i] = p[i].first;
                q[i + p.size()] = p[i].second;
            }

        } else {

            plhs[1] = mxCreateCellMatrix(maxLabel + 1, 1);
            for (int k = 0; k <= maxLabel; k++) {
                drwnPRCurve curve;
                for (unsigned i = 0; i < labels.size(); i++) {
                    vector<double> m(scores[i]);
                    drwn::expAndNormalize(m);

                    if (labels[i] == k) {
                        curve.accumulatePositives(m[k]);
                    } else if (labels[i] >= 0) {
                        curve.accumulateNegatives(m[k]);
                    }
                }
            
                vector<pair<double, double> > p = curve.getCurve();
                mxSetCell(plhs[1], k, mxCreateDoubleMatrix(p.size(), 2, mxREAL));

                double *q = mxGetPr(mxGetCell(plhs[1], k));
                for (unsigned i = 0; i < p.size(); i++) {
                    q[i] = p[i].first;
                    q[i + p.size()] = p[i].second;
                }
            }
        }
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexQPSolver.cpp
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
#include "drwnPGM.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: [x, J] = mexQPSolver(Q, p, r, [A, b, [G, h, [lb, ub, [x0,]]]] [options]);\n");
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

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (mxIsStruct(prhs[nrhs - 1])) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
        nrhs -= 1;
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    if ((nrhs < 3) || (nrhs == 4) || (nrhs == 6) || (nrhs == 8) || (nrhs > 10)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse problem
    MatrixXd P;
    VectorXd q;

    drwnMatlabUtils::mxArrayToEigen(prhs[0], P);
    drwnMatlabUtils::mxArrayToEigen(prhs[1], q);
    DRWN_ASSERT(mxGetNumberOfElements(prhs[2]) == 1);
    double r = mxGetScalar(prhs[2]);

    drwnLogBarrierQPSolver solver(P, q, r);

    if ((nrhs >= 5) && (!mxIsEmpty(prhs[3]) || !mxIsEmpty(prhs[4]))) {
        MatrixXd A;
        VectorXd b;
        drwnMatlabUtils::mxArrayToEigen(prhs[3], A);
        drwnMatlabUtils::mxArrayToEigen(prhs[4], b);
        solver.setEqConstraints(A, b);
    }

    if ((nrhs >= 7) && (!mxIsEmpty(prhs[5]) || !mxIsEmpty(prhs[6]))) {
        MatrixXd G;
        VectorXd h;
        drwnMatlabUtils::mxArrayToEigen(prhs[5], G);
        drwnMatlabUtils::mxArrayToEigen(prhs[6], h);
        solver.setIneqConstraints(G, h);
    }

    if ((nrhs >= 9) && (!mxIsEmpty(prhs[7]) || !mxIsEmpty(prhs[8]))) {
        VectorXd lb, ub;
        if (mxIsEmpty(prhs[7])) {
            lb = VectorXd::Constant(solver.size(), -DRWN_DBL_MAX);
        } else {
            drwnMatlabUtils::mxArrayToEigen(prhs[7], lb);
        }
        if (mxIsEmpty(prhs[8])) {
            ub = VectorXd::Constant(solver.size(), DRWN_DBL_MAX);
        } else {
            drwnMatlabUtils::mxArrayToEigen(prhs[8], ub);
        }
        solver.setBounds(lb, ub);
    }

    if (nrhs == 10) {
        VectorXd x0;
        drwnMatlabUtils::mxArrayToEigen(prhs[9], x0);
        solver.initialize(x0);
    } else {
        solver.findFeasibleStart();
    }

    // solve QP
    double J = solver.solve();

    if (nlhs > 0) {
        plhs[0] = mxCreateDoubleMatrix(solver.size(), 1, mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (int i = 0; i < solver.size(); i++) {
            p[i] = solver[i];
        }
    }

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        mxGetPr(plhs[1])[0] = J;
    }

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

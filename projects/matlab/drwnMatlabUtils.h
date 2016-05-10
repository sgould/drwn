/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <cassert>
#include <string>
#include <map>

// eigen
#include "Eigen/Core"

// matlab
#include "mex.h"
#include "matrix.h"

// darwin
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnPGM.h"

using namespace std;
using namespace Eigen;

// drwnMatlabUtils ---------------------------------------------------------
//! Utility functions for converting between Matlab and Darwin datatypes.

namespace drwnMatlabUtils {

    //! print a message to the matlab console
    void mexMsgTxt(const char *message) { mexPrintf("%s\n", message); }

    //! sets drwnLogger callbacks to output to Matlab
    void setupLoggerCallbacks();

    //! parse a table factor from a structure array \p s with fields
    //! \p s.vars and \p s.data
    drwnTableFactor *parseFactor(const drwnVarUniversePtr& pUniverse,
        const mxArray *factor, int factorIndx = 0);

    //! prints standard options to Matlab console
    void printStandardOptions();

    //! initializes standard options structure
    void initializeStandardOptions(map<string, string>& options);

    //! parses Matlab structure of options, assuming that each field is
    //! either a scalar or a string, and returns true if any options were set.
    bool parseOptions(const mxArray *optionStruct, map<string, string>& options);

    //! processes standard options
    void processStandardOptions(map<string, string>& options);

    //! converts a Matlab matrix to an stl vector of vectors
    void mxArrayToVector(const mxArray *m, vector<vector<double> > &v);

    //! converts a Matlab matrix to an eigen matrix
    void mxArrayToEigen(const mxArray *m, MatrixXd& A);

    //! converts a Matlab matrix to an eigen vector
    void mxArrayToEigen(const mxArray *m, VectorXd& b);

    //! converts a multi-dimensional Matlab matrix to a vector of eigen matrices
    void mxArrayToEigen(const mxArray *m, vector<MatrixXd>& A);
};

// drwnMatlabUtils implementation ------------------------------------------

// TODO: work out how to compile these not inline

void drwnMatlabUtils::setupLoggerCallbacks()
{
    // setup callbacks
    drwnLogger::showFatalCallback = mexErrMsgTxt;
    drwnLogger::showErrorCallback = mexErrMsgTxt;
    drwnLogger::showWarningCallback = mexWarnMsgTxt;
    drwnLogger::showStatusCallback = drwnMatlabUtils::mexMsgTxt;
    drwnLogger::showMessageCallback = drwnMatlabUtils::mexMsgTxt;

    // reset since statics persist between Matlab calls
    drwnLogger::setLogLevel(DRWN_LL_MESSAGE);
}


drwnTableFactor *drwnMatlabUtils::parseFactor(const drwnVarUniversePtr& pUniverse,
    const mxArray *factor, int factorIndx)
{
    mxAssert(mxIsStruct(factor), "invalid factor datastructure");

    // add variables
    mxArray *vars = mxGetField(factor, factorIndx, "vars");
    mxAssert(vars != NULL, "field 'vars' missing");

    drwnTableFactor *phi = new drwnTableFactor(pUniverse);
    for (int i = 0; i < mxGetNumberOfElements(vars); i++) {
        phi->addVariable((int)mxGetPr(vars)[i]);
    }

    // add data
    mxArray *data = mxGetField(factor, factorIndx, "data");
    if ((data == NULL) || mxIsEmpty(data)) {
        return phi;
    }
    mxAssert(mxGetNumberOfElements(data) == (int)phi->entries(),
        "wrong number of elements in field 'data'");

    for (int i = 0; i < phi->entries(); i++) {
        (*phi)[i] = mxGetPr(data)[i];
    }

    return phi;
}

void drwnMatlabUtils::printStandardOptions()
{
    mexPrintf("  config   :: configure Darwin from XML file\n");
    mexPrintf("  set      :: set configuration (module, property, value triplets)\n");
    mexPrintf("  profile  :: profile code\n");
    mexPrintf("  verbose  :: show verbose messages\n");
    mexPrintf("  debug    :: show debug messages\n");
}

void drwnMatlabUtils::initializeStandardOptions(map<string, string>& options)
{
    options[string("config")] = string("");
    options[string("set")] = string("");
    options[string("profile")] = string("0");
    options[string("verbose")] = string("0");
    options[string("debug")] = string("0");
}

bool drwnMatlabUtils::parseOptions(const mxArray *optionStruct, map<string, string>& options)
{
    if ((optionStruct == NULL) || (mxIsEmpty(optionStruct))) {
        return false;
    }

    if (!mxIsStruct(optionStruct)) {
        DRWN_LOG_FATAL("options must be a Matlab structure");
    }

    int N = mxGetNumberOfFields(optionStruct);
    for (int i = 0; i < N; i++) {
        // get field name and value
        const char *name = mxGetFieldNameByNumber(optionStruct, i);
        mxArray *value = mxGetFieldByNumber(optionStruct, 0, i);
        if ((value == NULL) || (mxIsEmpty(value))) {
            continue;
        }

        // check value type
        if (mxIsChar(value)) {
            char *v = mxArrayToString(value);
            options[string(name)] = string(v);
            mxFree(v);
        } else if (mxIsNumeric(value)) {
            if (mxGetNumberOfElements(value) == 1) {
                options[string(name)] = toString(mxGetScalar(value));
            } else {
                string v;
                for (int i = 0; i < mxGetNumberOfElements(value); i++) {
                    if (i > 0) v = v + string(" ");
                    v = v + toString(mxGetPr(value)[i]);
                }
                options[string(name)] = v;
            }
        } else {
            mexErrMsgTxt("invalid option value");
        }
    }

    return true;
}

void drwnMatlabUtils::processStandardOptions(map<string, string>& options)
{
    // xml configuration
    if (!options[string("config")].empty()) {
        drwnConfigurationManager::get().configure(options[string("config")].c_str());
    }

    // module property setting
    if (!options[string("set")].empty()) {
        vector<string> tokens;
        const int n = drwn::parseString(options[string("set")], tokens);
        if (n == 1) {
            drwnConfigurationManager::get().showModuleUsage(tokens[0].c_str());
            mexErrMsgTxt("");
        }
        if (n % 3 != 0) {
            DRWN_LOG_FATAL("'set' option requires triplets of the form 'module property value'");
        }

        for (int i = 0; i < n; i += 3) {
            drwnConfigurationManager::get().configure(tokens[i].c_str(),
                tokens[i + 1].c_str(), tokens[i + 2].c_str());
        }
    }

    // code profiling
    drwnCodeProfiler::enabled = (atoi(options[string("profile")].c_str()) != 0);

    // message verbosity
    if (atoi(options[string("verbose")].c_str()) != 0) {
        drwnLogger::setLogLevel(DRWN_LL_VERBOSE);
    }
    if (atoi(options[string("debug")].c_str()) != 0) {
        drwnLogger::setLogLevel(DRWN_LL_DEBUG);
    }
}

void drwnMatlabUtils::mxArrayToVector(const mxArray *m, vector<vector<double> > &v)
{
    if (mxGetClassID(m) != mxDOUBLE_CLASS) {
        DRWN_LOG_FATAL("expecting matrix of type double");
    }

    const int nRows = mxGetM(m); // number of rows
    const int nCols = mxGetN(m); // number of columns
    v.resize(nRows);
    const double *p = mxGetPr(m);
    for (int i = 0; i < nRows; i++) {
        v[i].resize(nCols);
        for (int j = 0; j < nCols; j++) {
            v[i][j] = p[j * nRows + i];
        }
    }
}

void drwnMatlabUtils::mxArrayToEigen(const mxArray *m, MatrixXd& A)
{
    if (mxGetClassID(m) != mxDOUBLE_CLASS) {
        DRWN_LOG_FATAL("expecting matrix of type double");
    }

    const int nRows = mxGetM(m); // number of rows
    const int nCols = mxGetN(m); // number of columns
    DRWN_LOG_DEBUG("parsing a " << nRows << "-by-" << nCols << " matrix...");

    A.resize(nRows, nCols);
    const double *p = mxGetPr(m);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            A(i, j) = p[j * nRows + i];
        }
    }
}

void drwnMatlabUtils::mxArrayToEigen(const mxArray *m, VectorXd& b)
{
    if (mxGetClassID(m) != mxDOUBLE_CLASS) {
        DRWN_LOG_FATAL("expecting matrix of type double");
    }

    const int nRows = mxGetM(m); // number of rows
    const int nCols = mxGetN(m); // number of columns

    b = Eigen::Map<VectorXd>(mxGetPr(m), nRows * nCols);
}

void drwnMatlabUtils::mxArrayToEigen(const mxArray *m, vector<MatrixXd>& A)
{
    if (mxGetNumberOfDimensions(m) < 3) {
        A.resize(1);
        mxArrayToEigen(m, A[0]);
        return;
    }

    DRWN_ASSERT(mxGetNumberOfDimensions(m) == 3);
    const int nRows = mxGetDimensions(m)[0]; // number of rows
    const int nCols = mxGetDimensions(m)[1]; // number of columns
    const int nChannels = mxGetDimensions(m)[2]; // number of channels
    DRWN_LOG_DEBUG("parsing a " << nRows << "-by-" << nCols << "-by-" << nChannels << " matrix...");

    A.resize(nChannels, MatrixXd::Zero(nRows, nCols));
    const double *p = mxGetPr(m);
    for (int c = 0; c < nChannels; c++) {
        for (int x = 0; x < nCols; x++) {
            for (int y = 0; y < nRows; y++) {
                A[c](y, x) = *p++;
            }
        }
    }
}

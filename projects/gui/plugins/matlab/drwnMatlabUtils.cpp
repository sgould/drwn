/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ Standard Libraries
#include <cstdlib>

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// Matlab library
#include "mat.h"

#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

Eigen::MatrixXd mat2eigen(const mxArray *m)
{
    DRWN_ASSERT(m != NULL);
    
    if (mxIsEmpty(m)) {
        return MatrixXd::Zero(0, 0);
    }

    DRWN_ASSERT(mxIsNumeric(m));
    return Eigen::Map<MatrixXd>(mxGetPr(m), mxGetM(m), mxGetN(m));
}

mxArray *eigen2mat(const Eigen::MatrixXd& m)
{
    mxArray *mData = mxCreateDoubleMatrix(m.rows(), m.cols(), mxREAL);
    memcpy((void *)mxGetPr(mData), (void *)m.data(),
        m.rows() * m.cols() * sizeof(double));
    return mData;
}

string mat2record(const mxArray *m, drwnDataRecord *r)
{
    DRWN_ASSERT((m != NULL) && mxIsStruct(m));
    DRWN_ASSERT(r != NULL);

    string key;
    mxArray *mKey = mxGetField(m, 0, "key");
    if (mKey == NULL) {
        DRWN_LOG_ERROR("Matlab structure is missing 'key' field");
    } else if (!mxIsEmpty(mKey)) {
        DRWN_ASSERT(mxIsChar(mKey));
        char *str = mxArrayToString(mKey);
        key = string(str);
        mxFree(str);
    }

    // TODO: structure

    mxArray *mData = mxGetField(m, 0, "data");
    if (mData == NULL) {
        DRWN_LOG_ERROR("Matlab structure is missing 'data' field");
        r->data() = MatrixXd::Zero(0, 0);
    } else {
        r->data() = mat2eigen(mData);
    }

    mxArray *mObjective = mxGetField(m, 0, "objective");
    if (mObjective == NULL) {
        DRWN_LOG_ERROR("Matlab structure is missing 'objective' field");
        r->objective() = VectorXd::Zero(0);
    } else {
        r->objective() = mat2eigen(mObjective);
    }

    mxArray *mGradient = mxGetField(m, 0, "gradient");
    if (mGradient == NULL) {
        DRWN_LOG_ERROR("Matlab structure is missing 'gradient' field");
        r->gradient() = MatrixXd::Zero(0, 0);
    } else {
        r->gradient() = mat2eigen(mGradient);
    }

    return key;
}

mxArray *record2mat(const drwnDataRecord *r, const string& key)
{
    DRWN_ASSERT(r != NULL);

    const char *fieldNames[5] = {"key", "structure", "data", "objective", "gradient"};
    mxArray *mStructure = mxCreateStructMatrix(1, 1, 5, fieldNames);
    DRWN_ASSERT(mStructure != NULL);

    // add fields
    mxSetField(mStructure, 0, "key", mxCreateString(key.c_str()));
    mxSetField(mStructure, 0, "structure", eigen2mat(r->structure().cast<double>()));
    mxSetField(mStructure, 0, "data", eigen2mat(r->data()));
    mxSetField(mStructure, 0, "objective", eigen2mat(r->objective()));
    mxSetField(mStructure, 0, "gradient", eigen2mat(r->gradient()));

    return mStructure;
}

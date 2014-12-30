/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    transform.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;

// main ----------------------------------------------------------------------

int main()
{
    // construct features vectors
    vector<vector<double> > x(100);
    for (unsigned i = 0; i < x.size(); i++) {
        x[i].resize(10);
        for (unsigned j = 0; j < x[i].size(); j++) {
            x[i][j] = (double)j + drand48();
        }
    }
    
    // learn normalization scale and offset
    drwnFeatureWhitener whitener;
    whitener.train(x);

    // normalize all feature vectors in-place
    whitener.transform(x);

    // construct new feature vector
    vector<double> y(10, 0);
    DRWN_LOG_MESSAGE("   original feature vector: " << toString(y));

    // apply previously learned transform
    whitener.transform(y);
    DRWN_LOG_MESSAGE("transformed feature vector: " << toString(y));

    return 0;
}

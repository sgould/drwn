/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    pca.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iomanip>

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
        DRWN_LOG_MESSAGE("x[" << std::setw(2) << i << "] = " << toString(x[i]));
    }
    
    // learn pca over raw features
    drwnPCA pca;
    pca.setProperty("outputDim", 5);
    pca.train(x);

    vector<vector<double> > y;
    pca.transform(x, y);
    for (unsigned i = 0; i < std::min<unsigned>(10, x.size()); i++) {
        DRWN_LOG_MESSAGE("y[" << std::setw(2) << i << "] = " << toString(y[i]));
    }

    // learn pca over quadratic features
    drwnPCA pca2;
    pca2.setProperty("outputDim", 5);
    pca2.train(x, drwnTFeatureMapTransform<drwnSquareFeatureMap>());
    pca2.transform(x, y, drwnTFeatureMapTransform<drwnSquareFeatureMap>());
    for (unsigned i = 0; i < std::min<unsigned>(10, x.size()); i++) {
        DRWN_LOG_MESSAGE("y[" << std::setw(2) << i << "] = " << toString(y[i]));
    }

    return 0;
}

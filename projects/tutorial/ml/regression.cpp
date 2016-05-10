/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    regression.cpp
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
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {0.5, 1.7, 2.3};

    // construct dataset
    drwnRegressionDataset dataset;
    for (int i = 0; i < 3; i++) {
        dataset.append(vector<double>(1, x[i]), y[i]);
    }

    // learn linear regression model
    drwnTLinearRegressor<drwnBiasFeatureMap> model(1);
    model.train(dataset);

    // show points from 0 to 5
    for (double i = 0.0; i <= 5.0; i += 1.0) {
        DRWN_LOG_MESSAGE("value at x=" << i << " is "
            << model.getRegression(vector<double>(1, i)));
    }

    return 0;
}

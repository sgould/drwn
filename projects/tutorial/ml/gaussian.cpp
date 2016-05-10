/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    gaussian.cpp
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
    // define mean and covariance
    VectorXd mu(2);
    mu << 0.5, 0.5;

    MatrixXd sigma(2, 2);
    sigma << 0.0964, -0.0505, -0.0505, 0.0540;

    // create gaussian P(x_1, x_2)
    drwnGaussian g(mu, sigma);

    // create conditional gaussian P(x_2 | x_1 = 1.0)
    map<int, double> x;
    x[0] = 1.0;
    drwnGaussian g2 = g.reduce(x);

    // show mean vector
    DRWN_LOG_MESSAGE("mean = " << g2.mean().transpose());

    return 0;
}

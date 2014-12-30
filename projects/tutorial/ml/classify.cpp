/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    classify.cpp
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
    // parameters
    const double SIGNAL2NOISE = 1.5;
    const int NUMPOSSAMPLES = 10;
    const int NUMNEGSAMPLES = 20;

    // construct dataset
    drwnClassifierDataset dataset;
    for (int i = 0; i < NUMPOSSAMPLES; i++) {
        dataset.append(vector<double>(1, SIGNAL2NOISE * 2.0 * (drand48() - 1.0)), 1);
    }
    for (int i = 0; i < NUMNEGSAMPLES; i++) {
        dataset.append(vector<double>(1, SIGNAL2NOISE * 2.0 * (drand48() - 1.0) + 1.0), 0);
    }

    // learn and evaluate different classifiers
    drwnClassifier *model;
    for (int i = 0; i < 3; i++) {
        // instantiate the classifier
        if (i == 0) {
            DRWN_LOG_MESSAGE("logistic classifier");
            model = new drwnTMultiClassLogistic<drwnBiasJointFeatureMap>(1, 2);
        } else if (i == 1) {
            DRWN_LOG_MESSAGE("boosted classifier");
            model = new drwnBoostedClassifier(1, 2);
        } else {
            DRWN_LOG_MESSAGE("decision-tree classifier");
            model = new drwnDecisionTree(1, 2);
        }

        // learn the paremeters
        model->train(dataset);

        // show confusion matrix (on the training set)
        drwnConfusionMatrix confusion(2, 2);
        confusion.accumulate(dataset, model);
        confusion.printCounts();

        delete model;
    }

    return 0;
}

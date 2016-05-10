/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    learnClassifier.cpp
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

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./learnClassifier [OPTIONS] <data> <labels>\n";
    cerr << "OPTIONS:\n"
         << "  -c <classifier>   :: classifier type (drwnBoostedClassifier, drwnDecisionTree,\n"
         << "                       drwnMultiClassLogistic (default), drwnRandomForest)\n"
         << "  -o <filename>     :: output model parameters\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *classifierName = "drwnMultiClassLogistic";
    const char *outputFilename = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-c", classifierName)
        DRWN_CMDLINE_STR_OPTION("-o", outputFilename)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    // read data and labels
    const char *dataFile = DRWN_CMDLINE_ARGV[0];
    const char *labelFile = DRWN_CMDLINE_ARGV[1];

    DRWN_LOG_MESSAGE("reading features vectors...");
    vector<vector<double> > features;
    ifstream ifs(dataFile);
    DRWN_ASSERT(!ifs.fail());

    // determine number of features
    int nFeatures = drwnCountFields(&ifs);

    // read feature vectors
    while (1) {
        vector<double> v(nFeatures);
        for (int i = 0; i < nFeatures; i++) {
            ifs >> v[i];
        }
        if (ifs.fail()) break;
        features.push_back(v);
    }
    ifs.close();
    DRWN_LOG_VERBOSE("..." << features.size()
        << " training examples of size " << nFeatures << " read");
    DRWN_ASSERT(!features.empty());

    // load training labels
    DRWN_LOG_MESSAGE("reading labels...");
    vector<int> labels(features.size(), -1);
    ifs.open(labelFile);
    DRWN_ASSERT(!ifs.fail());
    for (int i = 0; i < (int)labels.size(); i++) {
        ifs >> labels[i];
    }
    ifs.close();

    int nClasses = *max_element(labels.begin(), labels.end()) + 1;
    DRWN_LOG_VERBOSE("...number of labels is " << nClasses);
    DRWN_ASSERT(nClasses > 1);

    drwnClassifier *classifier = drwnClassifierFactory::get().create(classifierName);
    DRWN_ASSERT(classifier != NULL);

    // train classifier
    DRWN_LOG_MESSAGE("training classifier...");
    classifier->initialize(nFeatures, nClasses);
    classifier->train(features, labels);

    // evaluate on training set
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        vector<int> predictions;
        classifier->getClassifications(features, predictions);

        drwnConfusionMatrix confusion(classifier->numClasses(), classifier->numClasses());
        confusion.accumulate(labels, predictions);
        confusion.printCounts(cout, "training confusion matrix");
    }

    // save classifier parameters
    if (outputFilename != NULL) {
        DRWN_LOG_MESSAGE("saving learned parameters...");
        classifier->write(outputFilename);
    }

    delete classifier;

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

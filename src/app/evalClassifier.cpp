/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    evalClassifier.cpp
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
    cerr << "USAGE: ./evalClassifier [OPTIONS] <parameters> <data>\n";
    cerr << "OPTIONS:\n"
         << "  -g <labels>       :: ground-truth labels for evaluation\n"
         << "  -o <filename>     :: output predictions\n"
         << "  -s <filename>     :: output classifier scores\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *trueLabelsFilename = NULL;
    const char *outputFilename = NULL;
    const char *outputScoreFile = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-g", trueLabelsFilename)
        DRWN_CMDLINE_STR_OPTION("-o", outputFilename)
        DRWN_CMDLINE_STR_OPTION("-s", outputScoreFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    // read data and labels
    const char *modelFile = DRWN_CMDLINE_ARGV[0];
    const char *dataFile = DRWN_CMDLINE_ARGV[1];

    // read classifier
    DRWN_LOG_MESSAGE("reading classifier parameters...");
    drwnClassifier *classifier = drwnClassifierFactory::get().createFromFile(modelFile);
    DRWN_ASSERT(classifier != NULL);

    DRWN_LOG_MESSAGE("reading features vectors...");
    vector<vector<double> > features;
    ifstream ifs(dataFile);
    DRWN_ASSERT(!ifs.fail());

    // determine number of features
    int nFeatures = drwnCountFields(&ifs);
    DRWN_ASSERT(nFeatures == classifier->numFeatures());

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
        << " evaluation examples of size " << nFeatures << " read");
    DRWN_ASSERT(!features.empty());

    // predict labels
    DRWN_LOG_MESSAGE("predicting labels...");
    vector<int> predictions;
    classifier->getClassifications(features, predictions);

    if (outputFilename != NULL) {
        ofstream ofs(outputFilename);
        ofs << toString(predictions) << endl;
        ofs.close();
    }

    // output scores
    if (outputScoreFile != NULL) {
        ofstream ofs(outputScoreFile);
        vector<double> scores;
        for (unsigned i = 0; i < features.size(); i++) {
            classifier->getClassScores(features[i], scores);
            ofs << toString(scores) << "\n";
        }
        ofs.close();
    }

    // load true labels
    if (trueLabelsFilename != NULL) {
        DRWN_LOG_MESSAGE("reading true labels...");
        vector<int> labels(features.size(), -1);
        ifs.open(trueLabelsFilename);
        DRWN_ASSERT(!ifs.fail());
        for (int i = 0; i < (int)labels.size(); i++) {
            ifs >> labels[i];
        }
        ifs.close();

        drwnConfusionMatrix confusion(classifier->numClasses(), classifier->numClasses());
        confusion.accumulate(labels, predictions);
        confusion.printCounts(cout, "evaluation confusion matrix");
    }

    delete classifier;

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

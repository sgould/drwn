/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinClassifiers.cpp
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

// prototypes ----------------------------------------------------------------

void testClassifier(const char *testName, drwnClassifier *classifier,
    const drwnClassifierDataset& dataset, const char *tmpFilename, bool bKeepFile);
void writePRCurves(const char *testName, const drwnClassifier *classifier,
    const drwnClassifierDataset& dataset, const string& filename);

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinClassifiers [OPTIONS] <data> <labels>\n";
    cerr << "OPTIONS:\n"
         << "  -registry         :: shows registered classifiers\n"
         << "  -hasKey           :: data and label files contain key field\n"
         << "  -tmpfile <base>   :: temporary filebase for save/load test\n"
         << "  -keepfile         :: keep temporary file after test\n"
         << "  -prfile <base>    :: file for precision-recall curve\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    bool bHasKey = false;
    const char *tmpFilebase = NULL;
    const char *prFilebase = NULL;
    bool bKeepFile = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("-registry")
            drwnClassifierFactory::get().dump();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_BOOL_OPTION("-hasKey", bHasKey)
        DRWN_CMDLINE_STR_OPTION("-tmpfile", tmpFilebase)
        DRWN_CMDLINE_BOOL_OPTION("-keepfile", bKeepFile)
        DRWN_CMDLINE_STR_OPTION("-prfile", prFilebase)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    // read data and labels
    const char *dataFile = DRWN_CMDLINE_ARGV[0];
    const char *labelFile = DRWN_CMDLINE_ARGV[1];

    drwnClassifierDataset dataset;
    DRWN_LOG_MESSAGE("reading features vectors...");
    ifstream ifs(dataFile);
    DRWN_ASSERT(!ifs.fail());

    // determine number of features
    int nFeatures = drwnCountFields(&ifs);
    if (bHasKey) nFeatures -= 1;

    // read feature vectors
    while (1) {
        vector<double> v(nFeatures);
        if (bHasKey) {
            ifs.ignore(numeric_limits<streamsize>::max(), ' ');
        }
        for (int i = 0; i < nFeatures; i++) {
            ifs >> v[i];
        }
        if (ifs.fail()) break;
        dataset.features.push_back(v);
    }
    ifs.close();
    DRWN_LOG_VERBOSE("..." << dataset.features.size()
        << " instances of size " << nFeatures << " read");
    DRWN_ASSERT(!dataset.features.empty());

    // load training labels
    DRWN_LOG_MESSAGE("reading labels...");
    dataset.targets.resize(dataset.features.size(), -1);
    ifs.open(labelFile);
    DRWN_ASSERT(!ifs.fail());
    for (int i = 0; i < dataset.size(); i++) {
        if (bHasKey) {
            ifs.ignore(numeric_limits<streamsize>::max(), ' ');
        }
        ifs >> dataset.targets[i];
    }
    ifs.close();

    const int nClasses = *max_element(dataset.targets.begin(), dataset.targets.end()) + 1;
    DRWN_LOG_VERBOSE("...number of labels is " << nClasses);
    DRWN_ASSERT(nClasses > 1);

    drwnClassifier *classifier = NULL;
    string tmpFilename;

    const char *names[] = {"drwnMultiClassLogistic", "drwnDecisionTree",
                           "drwnBoostedClassifier", "drwnRandomForest", 
                           "drwnCompositeClassifier"};

    for (unsigned i = 0; i < sizeof(names) / sizeof(const char *); i++) {
        if (tmpFilebase) {
            tmpFilename = string(tmpFilebase) + string(".") +
                string(names[i]) + string(".xml");            
        }
        classifier = drwnClassifierFactory::get().create(names[i]);
        classifier->initialize(nFeatures, nClasses);

        testClassifier(names[i], classifier,
            dataset, tmpFilename.c_str(), bKeepFile);
        if (prFilebase != NULL) {
            writePRCurves(names[i], classifier, dataset,
                string(prFilebase) + string(".logistic"));
        }
        
        delete classifier;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// private helper functions --------------------------------------------------

void testClassifier(const char *testName, drwnClassifier *classifier,
    const drwnClassifierDataset& dataset, const char *tmpFilename, bool bKeepFile)
{
    DRWN_ASSERT(classifier != NULL);

    // test training and evaluation
    classifier->train(dataset.features, dataset.targets);

    vector<int> predictions;
    classifier->getClassifications(dataset.features, predictions);

    drwnConfusionMatrix confusion(classifier->numClasses(), classifier->numClasses());
    confusion.accumulate(dataset.targets, predictions);
    confusion.printCounts(cout, testName);

    // test i/o
    if ((tmpFilename != NULL) && (strlen(tmpFilename) > 0)) {
        classifier->write(tmpFilename);
        classifier->initialize(0, 0);
        classifier->read(tmpFilename);

        predictions.clear();
        classifier->getClassifications(dataset.features, predictions);
        drwnConfusionMatrix confusionValidation(classifier->numClasses(),
            classifier->numClasses());
        confusionValidation.accumulate(dataset.targets, predictions);
        confusionValidation.printCounts(cout, testName);

        if (!bKeepFile) {
            remove(tmpFilename);
        }
    }
}

void writePRCurves(const char *testName, const drwnClassifier *classifier,
    const drwnClassifierDataset& dataset, const string& filename)
{
    drwnPRCurve pr;

    const int nClasses = dataset.maxTarget() + 1;
    if (nClasses == 2) {
        pr.accumulate(dataset, classifier);
        pr.writeCurve((filename + string(".txt")).c_str());
    } else {

        // compute marginals (normalized score)
        vector<vector<double> > marginals;
        classifier->getClassMarginals(dataset.features, marginals);

        // compute pr curve for each class
        for (int c = 0; c < nClasses; c++) {
            pr.clear();
            for (int i = 0; i < dataset.size(); i++) {
                if (dataset.targets[i] == c) {
                    pr.accumulatePositives(marginals[i][c]);
                } else {
                    pr.accumulateNegatives(marginals[i][c]);
                }
            }
            pr.writeCurve((filename + string("_") + toString(c) + string(".txt")).c_str());
        }
    }
}

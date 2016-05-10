/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexEvalClassifier.cpp
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

// matlab headers
#include "mex.h"
#include "matrix.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

// project headers
#include "drwnMatlabUtils.h"

using namespace std;
using namespace Eigen;

void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: [predictions, scores] = mexEvalClassifier(classifier, features, [options]);\n");
    mexPrintf("  features :: N-by-D matrix of (transposed) feature vectors\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // set drwnLogger callbacks
    drwnMatlabUtils::setupLoggerCallbacks();

    // check function arguments
    if (nrhs == 0) {
        usage();
        return;
    }

    if ((nrhs < 2) || (nrhs > 3)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    if (nrhs == 3) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse and create classifier
    DRWN_ASSERT_MSG(mxIsChar(prhs[0]), "classifier must be an XML string");
    char *xmlString = mxArrayToString(prhs[0]);

    drwnXMLDoc xml;
    xml.parse<rapidxml::parse_no_data_nodes>(xml.allocate_string(xmlString));
    mxFree(xmlString);

    drwnXMLNode *node = xml.first_node();
    DRWN_ASSERT_MSG(node != NULL, "could not create classifier from xml string");
    drwnClassifier *classifier = drwnClassifierFactory::get().createFromXML(*node);
    DRWN_ASSERT_MSG(classifier != NULL, "could not create classifier from xml string");

    // parse features
    vector<vector<double> > features;
    drwnMatlabUtils::mxArrayToVector(prhs[1], features);

    // whiten features
    node = node->next_sibling();
    if (node != NULL) {
        drwnFeatureWhitener whitener;
        whitener.load(*node);
        whitener.transform(features);
    }

    // generate predictions and scores
    if (nlhs == 0) {
        DRWN_LOG_WARNING("not running classifier since no output arguments");
    } else if (nlhs == 1) {
        // output predictions
        plhs[0] = mxCreateDoubleMatrix(features.size(), 1, mxREAL);
        double *p = mxGetPr(plhs[0]);
        for (unsigned i = 0; i < features.size(); i++) {
            p[i] = (double)classifier->getClassification(features[i]);
        }
    } else {
        // output predictions and scores
        plhs[0] = mxCreateDoubleMatrix(features.size(), 1, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(features.size(), classifier->numClasses(), mxREAL);
        double *p = mxGetPr(plhs[0]);
        double *q = mxGetPr(plhs[1]);
        vector<double> scores;
        for (unsigned i = 0; i < features.size(); i++) {
            classifier->getClassScores(features[i], scores);
            for (unsigned j = 0; j < scores.size(); j++) {
                q[i + j * features.size()] = scores[j];
            }
            p[i] = drwn::argmax(scores);
        }
    }

    vector<int> predictions;
    classifier->getClassifications(features, predictions);

    // delete classifier
    delete classifier;

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

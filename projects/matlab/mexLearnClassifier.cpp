/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexLearnClassifier.cpp
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
    mexPrintf("USAGE: classifier = mexLearnClassifier(features, labels, [weights, [options]]);\n");
    mexPrintf("  features :: N-by-D matrix of (transposed) feature vectors\n");
    mexPrintf("  labels   :: N-vector of labels (0-based)\n");
    mexPrintf("  weights  :: N-vector of non-negative weights (or empty)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("  method   :: classifier type (default: 'drwnMultiClassLogistic')\n");
    mexPrintf("  whiten   :: whiten the features (default: 0)\n");
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

    if ((nrhs < 2) || (nrhs > 4)) {
        usage();
        mexErrMsgTxt("incorrect number of input arguments");
    }

    DRWN_ASSERT_MSG(mxGetM(prhs[1]) == mxGetM(prhs[0]), "mismatch between features and labels");
    DRWN_ASSERT_MSG((nrhs < 3) || (mxIsEmpty(prhs[2])) ||
        (mxGetM(prhs[2]) == mxGetM(prhs[0])), "mismatch between features and weights");

    // parse options
    map<string, string> options;
    drwnMatlabUtils::initializeStandardOptions(options);
    options[string("method")] = string("drwnMultiClassLogistic");
    options[string("whiten")] = string("0");
    if (nrhs == 4) {
        drwnMatlabUtils::parseOptions(prhs[nrhs - 1], options);
    }
    drwnMatlabUtils::processStandardOptions(options);
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("mex"));

    // parse dataset
    drwnClassifierDataset data;
    drwnMatlabUtils::mxArrayToVector(prhs[0], data.features);

    data.targets.resize(data.features.size());
    int maxLabel = 0;
    switch (mxGetClassID(prhs[1])) {
    case mxDOUBLE_CLASS:
        {
            const double *p = mxGetPr(prhs[1]);
            for (unsigned i = 0; i < data.targets.size(); i++) {
                data.targets[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    case mxINT32_CLASS:
        {
            const int32_t *p = (const int32_T *)mxGetData(prhs[1]);
            for (unsigned i = 0; i < data.targets.size(); i++) {
                data.targets[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    case mxLOGICAL_CLASS:
        {
            const mxLogical *p = (const mxLogical *)mxGetData(prhs[1]);
            for (unsigned i = 0; i < data.targets.size(); i++) {
                data.targets[i] = (int)p[i];
                maxLabel = std::max(maxLabel, (int)p[i]);
            }
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized datatype, try labels = double(labels);");
    }
    DRWN_ASSERT_MSG(maxLabel > 0, "classifier needs at least two labels");

    if ((nrhs > 2) && (!mxIsEmpty(prhs[2]))) {
        data.weights.resize(data.features.size());
        const double *p = mxGetPr(prhs[2]);
        for (unsigned i = 0; i < data.weights.size(); i++) {
            data.weights[i] = p[i];
        }
    }

    // check that all classes are present
    vector<int> histogram(maxLabel + 1, 0);
    for (unsigned i = 0; i < data.targets.size(); i++) {
        if (data.targets[i] < 0) continue;
        histogram[data.targets[i]] += 1;
    }
    for (unsigned i = 0; i < histogram.size(); i++) {
        if (histogram[i] == 0) {
            DRWN_LOG_FATAL("class label " << i << " has no examples");
        }
    }

    // whiten features
    drwnFeatureWhitener whitener;
    if (atoi(options[string("whiten")].c_str()) != 0) {
        DRWN_LOG_DEBUG("whitening " << data.numFeatures() << " features");
        whitener.train(data.features);
        whitener.transform(data.features);
    }

    // train classifier
    drwnClassifier *classifier = drwnClassifierFactory::get().create(options[string("method")].c_str());
    DRWN_ASSERT_MSG(classifier != NULL, "unknown classifier type \"" << options[string("method")] << "\"");

    DRWN_LOG_DEBUG("learning a " << (maxLabel + 1) << "-class classifier from "
        << data.size() << " features of length " << data.numFeatures());
    classifier->initialize(data.numFeatures(), maxLabel + 1);
    classifier->train(data);

    // evaluate on training set
    if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) {
        vector<int> predictions;
        classifier->getClassifications(data.features, predictions);

        drwnConfusionMatrix confusion(classifier->numClasses(), classifier->numClasses());
        confusion.accumulate(data.targets, predictions);
        confusion.printCounts(cout, "training confusion matrix");
    }

    // return classifier
    drwnXMLDoc xml;

    // xml declaration
    drwnXMLNode* decl = xml.allocate_node(rapidxml::node_declaration);
    decl->append_attribute(xml.allocate_attribute("version", "1.0"));
    decl->append_attribute(xml.allocate_attribute("encoding", "utf-8"));
    xml.append_node(decl);

    // root node (classifier)
    drwnXMLNode *node = xml.allocate_node(rapidxml::node_element,
        xml.allocate_string(classifier->type()));
    xml.append_node(node);
    classifier->save(*node);

    drwnAddXMLAttribute(*node, "drwnVersion", DRWN_VERSION, false);

    // add whitener
    if (atoi(options[string("whiten")].c_str()) != 0) {
        node = xml.allocate_node(rapidxml::node_element, whitener.type());
        xml.append_node(node);
        whitener.save(*node);
    }

    std::string buffer;
    print(std::back_inserter(buffer), xml, 0);
    if (nlhs == 1) {
        plhs[0] = mxCreateString(buffer.c_str());
    }

    delete classifier;

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
}

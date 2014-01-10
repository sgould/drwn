/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    learnPixelSegModel.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for learning parameters of the multi-class image labeling CRF.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// opencv library headers
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./learnPixelSegModel [OPTIONS] <trainList>\n";
    cerr << "OPTIONS:\n"
         << "  -component <str>  :: model component to learn BOOSTED (default),\n"
         << "                       UNARY, CONTRAST, ROBUSTPOTTS or CONTRASTANDPOTTS\n"
         << "  -subSample <n>    :: subsample data by 1/<n>\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *modelComponent = "BOOSTED";
    int subSample = 0;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-component", modelComponent)
        DRWN_CMDLINE_INT_OPTION("-subSample", subSample)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read list of training images
    const char *trainList = DRWN_CMDLINE_ARGV[0];

    DRWN_LOG_MESSAGE("Reading training list from " << trainList << "...");
    vector<string> baseNames = drwnReadFile(trainList);
    drwn::shuffle(baseNames); // permute names for better subsampling
    DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");

    // initialize model
    drwnPixelSegModel *model = new drwnPixelSegModel();
    string modelFilename = gMultiSegConfig.filebase("modelsDir", "pixelSegModel.xml");

    // learn boosted classifiers
    if (!strcasecmp(modelComponent, "BOOSTED")) {
        model->learnBoostedPixelModels(baseNames, subSample);
    }

    // learn pixel singletons
    if (!strcasecmp(modelComponent, "UNARY")) {
        model->read(modelFilename.c_str());
        model->learnPixelUnaryModel(baseNames, subSample);
    }

    // learn pixel pairwise contrast weight
    if (!strcasecmp(modelComponent, "CONTRAST")) {
        model->read(modelFilename.c_str());
        model->learnPixelContrastWeight(baseNames);
    }

    // learn robust potts weight
    if (!strcasecmp(modelComponent, "ROBUSTPOTTS")) {
        model->read(modelFilename.c_str());
        model->learnRobustPottsWeight(baseNames);
    }

    // learn contrast and robust potts weights jointly
    if (!strcasecmp(modelComponent, "CONTRASTANDPOTTS")) {
        model->read(modelFilename.c_str());
        model->learnPixelContrastAndRobustPottsWeights(baseNames);
    }

    // save the learned model
    model->write(modelFilename.c_str());
    delete model;

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

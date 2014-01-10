/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    scorePixelLabels.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for scoring pixel labelling MRF results.
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
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./scorePixelLabels [OPTIONS] <evalList>\n";
    cerr << "OPTIONS:\n"
         << "  -inLabels <ext>   :: extension for label input (default: .txt)\n"
         << "  -outScores <file> :: output filename for scored results\n"
         << "  -confusion        :: show entire confusion matrix\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *inLabelExt = ".txt";
    const char *outScoreFile = NULL;
    bool bShowConfusion = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-inLabels", inLabelExt)
        DRWN_CMDLINE_STR_OPTION("-outScores", outScoreFile)
        DRWN_CMDLINE_BOOL_OPTION("-confusion", bShowConfusion)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read list of evaluation images
    const char *evalList = DRWN_CMDLINE_ARGV[0];

    DRWN_LOG_MESSAGE("Reading evaluation list from " << evalList << "...");
    vector<string> baseNames = drwnReadFile(evalList);
    DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");

    const int nLabels = gMultiSegRegionDefs.maxKey() + 1;
    drwnConfusionMatrix confusion(nLabels);
    vector<double> scores(baseNames.size());

    // process results
    DRWN_LOG_MESSAGE("Processing results (" << inLabelExt << ")...");
    int hProcessImage = drwnCodeProfiler::getHandle("processImage");
    for (int i = 0; i < (int)baseNames.size(); i++) {
        drwnCodeProfiler::tic(hProcessImage);
        string lblFilename = gMultiSegConfig.filename("lblDir", baseNames[i], "lblExt");
        string resFilename = gMultiSegConfig.filebase("outputDir", baseNames[i]) + string(inLabelExt);
        DRWN_LOG_STATUS("..." << baseNames[i] << " (" << (i + 1) << " of "
            << baseNames.size() << ")");

        // read ground-truth labels
        MatrixXi actualLabels;
        //drwnReadUnknownMatrix(actualLabels, lblFilename.c_str());
        drwnLoadPixelLabels(actualLabels, lblFilename.c_str(), nLabels);

        // read inferred labels
        MatrixXi predictedLabels(actualLabels.rows(), actualLabels.cols());
        drwnReadMatrix(predictedLabels, resFilename.c_str());

        DRWN_ASSERT((predictedLabels.rows() == actualLabels.rows()) &&
            (predictedLabels.cols() == actualLabels.cols()));

        // accumulate results for this image
        drwnConfusionMatrix imageConfusion(nLabels);
        for (int y = 0; y < actualLabels.rows(); y++) {
            for (int x = 0; x < actualLabels.cols(); x++) {
                if (actualLabels(y, x) < 0) continue;
                imageConfusion.accumulate(actualLabels(y, x), predictedLabels(y, x));
            }
        }

        scores[i] = imageConfusion.accuracy();

        // add to dataset results
        confusion.accumulate(imageConfusion);

        drwnCodeProfiler::toc(hProcessImage);
    }

    // display results
    if (bShowConfusion) {
        confusion.printRowNormalized(cout, "--- Class Confusion Matrix ---");
        confusion.printPrecisionRecall(cout, "--- Recall/Precision (by Class) ---");
        confusion.printF1Score(cout, "--- F1-Score (by Class) ---");
        confusion.printJaccard(cout, "--- Intersection/Union Metric (by Class) ---");
    }

    DRWN_LOG_MESSAGE("Overall class accuracy: " << confusion.accuracy() << " (" << evalList << ")");
    DRWN_LOG_MESSAGE("Average class accuracy: " << confusion.avgRecall() << " (" << evalList << ")");
    DRWN_LOG_MESSAGE("Average jaccard score:  " << confusion.avgJaccard() << " (" << evalList << ")");

    // write scores
    if (outScoreFile != NULL) {
        ofstream ofs(outScoreFile);
        DRWN_ASSERT_MSG(!ofs.fail(), outScoreFile);
        for (int i = 0; i < (int)scores.size(); i++) {
            ofs << scores[i] << "\n";
        }
        ofs.close();
    }

    // clean up and print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

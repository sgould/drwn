/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    convertFromLabelMe.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Application for converting LabelMe XML format to (colour) annotated images.
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
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

#define WINDOW_NAME "convertFromLabelMe"

using namespace std;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./convertFromLabelMe [OPTIONS] (<imageList>|<baseName>)\n";
    cerr << "OPTIONS:\n"
         << "  -tags <filename>  :: load synonym list\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *tagFilename = NULL;
    bool bVisualize = false;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-tags", tagFilename)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 1) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // read list of evaluation images
    const char *imageList = DRWN_CMDLINE_ARGV[0];

    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }

    // check for input directory
    if (!drwnDirExists(gMultiSegConfig.filebase("lblDir", "").c_str())) {
        DRWN_LOG_FATAL("input/output labels directory " << gMultiSegConfig.filebase("lblDir", "")
            << " does not exist");
    }

    // read list of tags
    map<string, string> tags;
    if (tagFilename != NULL) {
        string latestTag;
        vector<string> lines = drwnReadLines(tagFilename);
        for (unsigned i = 0; i < lines.size(); i++) {
            if (lines.size() < 4) continue;
            string head = lines[i].substr(0, 4);
            string tail = lines[i].substr(5, lines[i].size() - 6);
            if (head == string("TAG:")) {
                latestTag = tail;
            } else {
                tags[tail] = latestTag;
                DRWN_LOG_DEBUG(tail << " maps to " << latestTag);
            }
        }
        DRWN_LOG_VERBOSE(tags.size() << " tags synonyms read");
    }

    // construct name-to-colour table
    map<string, unsigned int> table;

    set<int> keys(gMultiSegRegionDefs.keys());
    for (set<int>::const_iterator ik = keys.begin(); ik != keys.end(); ++ik) {
        table[gMultiSegRegionDefs.name(*ik)] = gMultiSegRegionDefs.color(*ik);
    }

    for (map<string, unsigned int>::const_iterator it = table.begin(); it != table.end(); ++it) {
        DRWN_LOG_DEBUG("colour (" << (int)gMultiSegRegionDefs.red(it->second) << ", "
            << (int)gMultiSegRegionDefs.green(it->second) << ", "
            << (int)gMultiSegRegionDefs.blue(it->second)
            << ") corresponds to label " << it->first);
    }

    // iterate over images
    for (int n = 0; n < (int)baseNames.size(); n++) {
        DRWN_LOG_VERBOSE("processing " << baseNames[n] << "...");

        // load image
        const string imgFilename = gMultiSegConfig.filename("imgDir", baseNames[n], "imgExt");
        cv::Mat img = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);

        // show image
        if (bVisualize) {
            cv::namedWindow(WINDOW_NAME);
            cv::imshow(WINDOW_NAME, img);
            cv::waitKey(100);
        }

        // create label image
        cv::Mat labels(img.rows, img.cols, CV_8UC3);

        // load annotation
        const string xmlFilename = gMultiSegConfig.filebase("lblDir", baseNames[n]) + string(".xml");

        drwnXMLDoc doc;
        drwnXMLNode *xml = drwnParseXMLFile(doc, xmlFilename.c_str(), "annotation");
        for (drwnXMLNode *node = xml->first_node("object"); node != NULL;
             node = node->next_sibling("object")) {
            drwnXMLNode *child = node->first_node("name");
            if ((child == NULL) || (child->value() == NULL)) {
                DRWN_LOG_WARNING("object has no name field");
                continue;
            }
            
            string name = string(child->value());
            if (tags.find(name) != tags.end()) {
                name = tags[name];
            }

            unsigned int colourIndex = 0;
            if (table.find(name) != table.end()) {
                colourIndex = table[name];
            } else {
                DRWN_LOG_WARNING("could not find " << name << " in colour table");
                continue;
            }
            cv::Scalar colour = CV_RGB(gMultiSegRegionDefs.red(colourIndex),
                gMultiSegRegionDefs.green(colourIndex), gMultiSegRegionDefs.blue(colourIndex));

            child = node->first_node("polygon");
            DRWN_ASSERT(child != NULL);
            const int numPoints = drwnCountXMLChildren(*child, "pt");
            DRWN_LOG_DEBUG(name << " (" << colourIndex << ") " << " has " << numPoints << " points");

            cv::Point* poly = new cv::Point[numPoints + 1];
            drwnXMLNode *pt = NULL;
            for (int j = 0; j < numPoints; j++) {
                pt = (j == 0) ? child->first_node("pt") : pt->next_sibling("pt");
                DRWN_ASSERT(pt != NULL);
                const int x = atoi(pt->first_node("x")->value());
                const int y = atoi(pt->first_node("y")->value());
                poly[j] = cv::Point(x, y);
            }
            poly[numPoints] = poly[0];

            cv::fillConvexPoly(labels, poly, (int)numPoints + 1, colour);
            delete[] poly;
        }

        if (bVisualize) {
            drwnShowDebuggingImage(labels, string(WINDOW_NAME), false);
        }

        // save labels
        const string lblFilename = gMultiSegConfig.filebase("lblDir", baseNames[n]) + string(".png");
        cv::imwrite(lblFilename, labels);
    }

    // wait for keypress if only image
    if (bVisualize && (baseNames.size() == 1)) {
        cv::waitKey(-1);
    }

    // clean up and print profile information
    cv::destroyAllWindows();
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

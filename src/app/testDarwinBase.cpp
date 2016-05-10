/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinBase.cpp
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

using namespace std;
using namespace Eigen;

// prototypes ----------------------------------------------------------------

void testBitArray();
void testFileUtils();
void testOrderedMap();
void testXMLUtils();
void testThreading();
void testStatsUtils();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinBase [OPTIONS] ((<test>)* | all)\n";
    cerr << "OPTIONS:\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "TESTS:\n"
         << "  bitarray          :: test drwnBitArray class\n"
         << "  fileutils         :: test drwnFileUtils functions\n"
         << "  orderedmap        :: test drwnOrderedMap template\n"
         << "  xmlutils          :: test drwnXMLUtils class\n"
         << "  threading         :: test drwnThreadPool class\n"
         << "  statsutils        :: test drwnStatsUtils functions\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
	DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("all")
            testBitArray();
            testFileUtils();
            testOrderedMap();
            testXMLUtils();
			testThreading();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("bitarray")   testBitArray();   DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("fileutils")  testFileUtils();  DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("orderedmap") testOrderedMap(); DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("xmlutils")   testXMLUtils();   DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("threading")  testThreading();  DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("statsutils") testStatsUtils(); DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// testBitArray --------------------------------------------------------------

void testBitArray()
{
    DRWN_LOG_MESSAGE("testing drwnBitArray...");
    for (int i = 12; i < 1024; i += 17) {
        drwnBitArray barray(i);
        barray.zeros();
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray.count() == 0);

        barray.set(3);
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray.count() == 1);

        drwnBitArray barray2 = barray;
        DRWN_ASSERT(barray == barray2);

        barray.set(5);
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray.count() == 2);
        DRWN_ASSERT(!(barray == barray2));

        barray.negate();
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray.count() == i - 2);

        barray2.bitwiseor(barray);
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray2.count() == i - 1);

        barray.ones();
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT_MSG(barray.count() == i, barray.count());

        barray2.bitwisexor(barray);
        if ((drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE) && (i == 12))
            barray.print();
        DRWN_ASSERT(barray2.count() == 1);
    }
    DRWN_LOG_MESSAGE("...passed");
}

// testFileUtils -----------------------------------------------------------------

void testFileUtils()
{
    DRWN_LOG_MESSAGE("testing drwnFileUtils...");

    string dir = drwnGetCurrentDir();
    int dirSize = drwnDirSize(dir.c_str());
    DRWN_LOG_VERBOSE("current directory " << dir << " has " << dirSize << " files");
    DRWN_ASSERT(drwnPathExists(dir.c_str()));
    DRWN_ASSERT(drwnDirExists(dir.c_str()));
    DRWN_ASSERT(!drwnFileExists(dir.c_str()));

    vector<string> dirContents = drwnDirectoryListing(dir.c_str());
    DRWN_ASSERT(dirContents.size() == (unsigned)dirSize);

    DRWN_LOG_MESSAGE("...passed");
}

// testOrderedMap ----------------------------------------------------------------

void testOrderedMap()
{
    DRWN_LOG_MESSAGE("testing drwnOrderedMap...");

    {
        drwnOrderedMap<string, int> map;
        map.insert(string("first string"), 100);
        map.insert(string("second string"), 20);
        map.insert(string("third string"), 300);

        DRWN_ASSERT(map.size() == 3);

        DRWN_ASSERT(map[string("first string")] == 100);
        DRWN_ASSERT(map[string("second string")] == 20);
        DRWN_ASSERT(map[string("third string")] == 300);

        DRWN_ASSERT(map[0] == 100);
        DRWN_ASSERT(map[1] == 20);
        DRWN_ASSERT(map[2] == 300);

        map[string("second string")] = 250;
        DRWN_ASSERT(map[1] == 250);

        map.erase(string("second string"));
        DRWN_ASSERT(map.size() == 2);

        DRWN_ASSERT(map[string("first string")] == 100);
        DRWN_ASSERT(map[string("third string")] == 300);

        DRWN_ASSERT(map[0] == 100);
        DRWN_ASSERT(map[1] == 300);
    }

    DRWN_LOG_MESSAGE("...passed");
}

// testXMLUtils ------------------------------------------------------------------

class MyObject {
public:
    VectorXd x;

    MyObject() { }
    MyObject(const VectorXd& ix) : x(ix) { }
    ~MyObject() { }

    void save(drwnXMLNode& xml) const { drwnXMLUtils::serialize(xml, x); }
    void load(drwnXMLNode& xml) { drwnXMLUtils::deserialize(xml, x); }
};

void testXMLUtils()
{
    DRWN_LOG_MESSAGE("testing drwnXMLUtils...");

    // write and read back a vector
    for (int n = 0; n < 5; n++) {
        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "vectorReadbackTest", NULL, false);

        VectorXd x(VectorXd::Random(n + 1));
        drwnXMLUtils::serialize(*node, x);
        if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE)
            drwnXMLUtils::dump(*node);
        VectorXd y;
        drwnXMLUtils::deserialize(*node, y);
        DRWN_LOG_VERBOSE("readback error = " << (x - y).transpose());
        DRWN_ASSERT_MSG(x == y, (x - y).transpose());
    }

#if 1
    // write and read back container of objects
    {
        vector<MyObject> container;
        for (int i = 0; i < 5; i++) {
            container.push_back(MyObject(VectorXd::Random(50)));
        }

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "containerReadbackTest", NULL, false);

        drwnXMLUtils::save(*node, "vector", container);
        if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE)
            drwnXMLUtils::dump(*node);

        vector<MyObject> container2;
        drwnXMLUtils::load(*node, "vector", container2);
        DRWN_ASSERT(container2.size() == container.size());

        for (unsigned i = 0; i < container.size(); i++) {
            DRWN_LOG_VERBOSE("readback error = " << (container[i].x - container2[i].x).transpose());
            DRWN_ASSERT_MSG(container[i].x == container2[i].x,
                (container[i].x - container2[i].x).transpose());
        }
    }
#endif

#if 1
    // write and read back container of object pointers
    {
        vector<MyObject *> container;
        for (int i = 0; i < 5; i++) {
            container.push_back(new MyObject(VectorXd::Random(50)));
        }

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, "containerReadbackTest", NULL, false);

        drwnXMLUtils::save(*node, "vector", container);
        if (drwnLogger::getLogLevel() >= DRWN_LL_VERBOSE)
            drwnXMLUtils::dump(*node);

        vector<MyObject *> container2;
        drwnXMLUtils::load(*node, "vector", container2);
        DRWN_ASSERT(container2.size() == container.size());

        for (unsigned i = 0; i < container.size(); i++) {
            DRWN_LOG_VERBOSE("readback error = " << (container[i]->x - container2[i]->x).transpose());
            DRWN_ASSERT_MSG(container[i]->x == container2[i]->x,
                (container[i]->x - container2[i]->x).transpose());

            delete container2[i];
            delete container[i];
        }
    }
#endif

    DRWN_LOG_MESSAGE("...passed");
}

// testThreading -----------------------------------------------------------------

class FibbonacciThreadJob : public drwnThreadJob {
protected:
	unsigned _n;
public:
    unsigned result;
  public:
    FibbonacciThreadJob(unsigned n) : _n(n), result(0) { /* do nothing */ }

    void operator()() {
		result = 0;

		// lets do some useless work (drand48() should round to zero)
		for (unsigned do_some_work = 0; do_some_work < 1e7; do_some_work++) {
			result = result + (int)drand48();
		}

		if (_n != 0) {
			unsigned f_prev = 0;
			result = 1;
			for (unsigned i = 1; i < _n; i++) {
				f_prev += result;
				std::swap(f_prev, result);
			}
		}

		// print out result when finished
		lock();
		cout << "...finished computing F(" << _n << ")" << endl;
		unlock();
	}
};

void testThreading()
{
	// start thread pool and add jobs
	drwnThreadPool threadPool;
	vector<FibbonacciThreadJob *> jobs;

	threadPool.start();
	for (unsigned i = 0; i <= 10; i++) {
            jobs.push_back(new FibbonacciThreadJob(i));
            threadPool.addJob(jobs.back());
	}

	// wait for jobs to finish (and show progress)
	threadPool.finish(true);

	// extract results (and delete jobs)
	for (unsigned i = 0; i < jobs.size(); i++) {
            cout << "F(" << i << ") = " << jobs[i]->result << endl;
            delete jobs[i];
	}
}

// testStatsUtils ----------------------------------------------------------------

void testStatsUtils()
{
    cout << toString(drwn::linSpaceVector(0.0, 10.0, 11)) << endl;
    cout << toString(drwn::logSpaceVector(1.0, 128.0, 8)) << endl;
    cout << toString(drwn::logSpaceVector(1.0, 128.0, 15)) << endl;
}

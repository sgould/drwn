/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testDarwinML.cpp
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

void testRegression();
void testSparseVector();
void testMaxSpeed();
void testFastExp();
void testDotProduct();
void testFeatureWhitener();
void testSuffStats();
void testPCA();
void testFisherLDA();
void testKMeans();
void testGaussianSampling();
void testGaussianMixture();
void testPRCurve();
void testCrossValidator();
void testQPSolver(const char *filename, bool bLogBarrier);
void testLPSolver();

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testDarwinML [OPTIONS] (<test>)*\n";
    cerr << "OPTIONS:\n"
         << "  -dump             :: dump classifier factory\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << "TESTS:\n"
         << "  regression        :: linear and quadratic regression tests\n"
         << "  sparseVector      :: sparse vector tests\n"
         << "  maxSpeed          :: speed test for maximization\n"
         << "  fastExp           :: test fast exponentiation\n"
         << "  dotProduct        :: test dot product\n"
         << "  featureWhitener   :: test feature whitener\n"
         << "  suffStats         :: test sufficient statistics\n"
         << "  pca               :: test principal component analysis\n"
         << "  lda               :: test fisher's linear discriminant analysis\n"
         << "  kmeans            :: test k-means clustering\n"
         << "  gaussianSampling  :: test gaussian sampling\n"
         << "  gaussianMixture   :: test gaussian mixture model\n"
         << "  pr                :: test precision-recall curve\n"
         << "  cv                :: test cross-validator\n"
         << "  qp <filename>     :: test QP solver (using XML file)\n"
         << "  lbqp <filename>   :: test log-barrier QP solver (using XML file)\n"
         << "  lp                :: test LP solver\n"
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_FLAG_BEGIN("-dump")
            drwnClassifierFactory::get().dump();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("regression")
            testRegression();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("sparseVector")
            testSparseVector();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("maxSpeed")
            testMaxSpeed();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("fastExp")
            testFastExp();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("dotProduct")
            testDotProduct();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("featureWhitener")
            testFeatureWhitener();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("suffStats")
            testSuffStats();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("pca")
            testPCA();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("lda")
            testFisherLDA();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("kmeans")
            testKMeans();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("gaussianSampling")
            testGaussianSampling();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("gaussianMixture")
            testGaussianMixture();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("pr")
            testPRCurve();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_FLAG_BEGIN("cv")
            testCrossValidator();
        DRWN_CMDLINE_FLAG_END
        DRWN_CMDLINE_OPTION_BEGIN("qp", p)
            testQPSolver(p[0], false);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_OPTION_BEGIN("lbqp", p)
            testQPSolver(p[0], true);
        DRWN_CMDLINE_OPTION_END(1)
        DRWN_CMDLINE_FLAG_BEGIN("lp")
            testLPSolver();
        DRWN_CMDLINE_FLAG_END
    DRWN_END_CMDLINE_PROCESSING(usage());

    if (DRWN_CMDLINE_ARGC != 0) {
        usage();
        return -1;
    }

    // print profile information
    drwnCodeProfiler::print();
    return 0;
}

// regression tests ----------------------------------------------------------

void testRegression()
{
    DRWN_FCN_TIC;

    const unsigned NUM_SAMPLES = 100000;
    const unsigned VEC_SIZE = 10;
    const double NOISE = 0.1;

    vector<vector<double> > x(NUM_SAMPLES, vector<double>(VEC_SIZE));
    vector<double> y(NUM_SAMPLES, 0.0);

    vector<double> theta(VEC_SIZE + 1);

    // generate parameters
    srand48(0);
    for (unsigned i = 0; i < VEC_SIZE + 1; i++) {
        theta[i] = (double)i + drand48();
    }

    // generate data
    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        for (unsigned j = 0; j < VEC_SIZE; j++) {
            x[i][j] = 2.0 * drand48() - 1.0;
            y[i] += theta[j] * (x[i][j] + 2.0 * NOISE * (drand48() - 0.5));
        }
        y[i] += theta[VEC_SIZE];
    }

    // test linear regression
    {
        DRWN_LOG_MESSAGE("testing drwnTLinearRegressor<drwnBiasFeatureMap>...");
        drwnTLinearRegressor<drwnBiasFeatureMap> model(VEC_SIZE);
        model.setProperty("penalty", 1.0e-6);
        model.train(x, y);

        // display learned parameters
        double bias = model.getRegression(vector<double>(VEC_SIZE, 0.0));
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            vector<double> z(VEC_SIZE, 0.0);
            z[i] = 1.0;
            double t = model.getRegression(z) - bias;
            cout << "\t" << setprecision(8) << t
                 << "\t" << setprecision(8) << theta[i]
                 << "\t" << setprecision(8) << fabs(theta[i] - t)
                 << "\t" << fabs(theta[i] - t) / theta[i] << "\n";
            DRWN_ASSERT(fabs(theta[i] - t) / (theta[i] + 1.0) < 0.1);
        }

        drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, model.type());
        model.save(*node);
        drwnXMLUtils::dump(*node);
    }

    // generate squared dependencies
    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        for (unsigned j = 0; j < VEC_SIZE; j++) {
            x[i][j] = 2.0 * drand48() - 1.0;
            y[i] += theta[j] * (x[i][j] * x[i][j] + 2.0 * NOISE * (drand48() - 0.5));
        }
        y[i] += theta[VEC_SIZE];
    }

    // test linear regression
    {
        DRWN_LOG_MESSAGE("testing drwnTLinearRegressor<drwnBiasFeatureMap>...");
        drwnTLinearRegressor<drwnBiasFeatureMap> model(VEC_SIZE);
        model.setProperty("penalty", 1.0e-6);
        model.train(x, y);

        // display learned parameters
        double bias = model.getRegression(vector<double>(VEC_SIZE, 0.0));
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            vector<double> z(VEC_SIZE, 0.0);
            z[i] = 1.0;
            double t = model.getRegression(z) - bias;
            cout << "\t" << setprecision(8) << t
                 << "\t" << setprecision(8) << theta[i]
                 << "\t" << setprecision(8) << fabs(theta[i] - t)
                 << "\t" << fabs(theta[i] - t) / theta[i] << "\n";
        }

        drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, model.type());
        model.save(*node);
        drwnXMLUtils::dump(*node);
    }

    // test square regression
    {
        DRWN_LOG_MESSAGE("testing drwnTLinearRegressor<drwnSquareFeatureMap>...");
        drwnTLinearRegressor<drwnSquareFeatureMap> model(VEC_SIZE);
        model.setProperty("penalty", 1.0e-6);
        model.train(x, y);

        // display learned parameters
        double bias = model.getRegression(vector<double>(VEC_SIZE, 0.0));
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            vector<double> z(VEC_SIZE, 0.0);
            z[i] = 1.0;
            double t = model.getRegression(z) - bias;
            cout << "\t" << setprecision(8) << t
                 << "\t" << setprecision(8) << theta[i]
                 << "\t" << setprecision(8) << fabs(theta[i] - t)
                 << "\t" << fabs(theta[i] - t) / theta[i] << "\n";
            DRWN_ASSERT(fabs(theta[i] - t) / (theta[i] + 1.0) < 0.1);
        }

        drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, model.type());
        model.save(*node);
        drwnXMLUtils::dump(*node);
    }

    // test quadratic regression
    {
        DRWN_LOG_MESSAGE("testing drwnTLinearRegressor<drwnQuadraticFeatureMap>...");
        drwnTLinearRegressor<drwnQuadraticFeatureMap> model(VEC_SIZE);
        model.setProperty("penalty", 1.0e-6);
        model.train(x, y);

        // display learned parameters
        double bias = model.getRegression(vector<double>(VEC_SIZE, 0.0));
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            vector<double> z(VEC_SIZE, 0.0);
            z[i] = 1.0;
            double t = model.getRegression(z) - bias;
            cout << "\t" << setprecision(8) << t
                 << "\t" << setprecision(8) << theta[i]
                 << "\t" << setprecision(8) << fabs(theta[i] - t)
                 << "\t" << fabs(theta[i] - t) / theta[i] << "\n";
            DRWN_ASSERT(fabs(theta[i] - t) / (theta[i] + 1.0) < 0.1);
        }

        drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;

        drwnXMLDoc xml;
        drwnXMLNode *node = drwnAddXMLChildNode(xml, model.type());
        model.save(*node);
        drwnXMLUtils::dump(*node);
    }

    DRWN_FCN_TOC;
}

void testSparseVector()
{
    DRWN_FCN_TIC;
    const unsigned NUM_TESTS = 10;
    const unsigned VEC_SIZE = 1000;

    // basic assignment
    DRWN_LOG_MESSAGE("testing operator[] for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = xs[i] = x_i;
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    // push_back
    DRWN_LOG_MESSAGE("testing push_back for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd;
        drwnSparseVec<double> xs;
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd.push_back(x_i);
            xs.push_back(x_i);
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    // insert
    DRWN_LOG_MESSAGE("testing insert for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd;
        drwnSparseVec<double> xs;
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd.insert(xd.end(), x_i);
            xs.insert(xs.end(), x_i);
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd;
        drwnSparseVec<double> xs;
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd.insert(xd.end(), x_i);
        }
        xs.insert(xs.end(), xd.begin(), xd.end());

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

#if 0
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd;
        drwnSparseVec<double> xs;
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd.insert(xd.begin(), x_i);
            xs.insert(xs.begin(), x_i);
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }
#endif

    // operators
    DRWN_LOG_MESSAGE("testing operator+= for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = x_i;
            xs[i] += x_i;
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    DRWN_LOG_MESSAGE("testing operator*= for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = xs[i] = x_i;
            xd[i] *= xd[i];
            xs[i] *= xs[i];
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    DRWN_LOG_MESSAGE("testing operator*=0.0 for drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = xs[i] = x_i;
            xd[i] *= 0.0;
            xs[i] *= 0.0;
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

    // timing test
    DRWN_LOG_MESSAGE("testing timing of drwnSparseVector class...");
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("std::vector"));
    for (unsigned t = 0; t < 100 * NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = x_i;
        }
    }
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("std::vector"));

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("drwnSparseVector"));
    for (unsigned t = 0; t < 100 * NUM_TESTS; t++) {
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < VEC_SIZE; i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xs[i] = x_i;
        }
    }
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("drwnSparseVector"));

    // iterator test
    DRWN_LOG_MESSAGE("testing drwnSparseVector::iterator class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = x_i;
        }

        vector<double>::const_iterator jt(xd.begin());
        for (drwnSparseVec<double>::iterator it = xs.begin(); it != xs.end(); ) {
            *it++ = *jt++;
        }

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }

#if 0
    //!\todo does sorting a sparse vector even make sense?

    // sort
    DRWN_LOG_MESSAGE("testing sort on drwnSparseVector class...");
    for (unsigned t = 0; t < NUM_TESTS; t++) {
        vector<double> xd(VEC_SIZE, 0.0);
        drwnSparseVec<double> xs(VEC_SIZE);
        for (unsigned i = 0; i < xd.size(); i++) {
            double x_i = drand48();
            if (drand48() > (double)t / (double)NUM_TESTS) x_i = 0.0;
            xd[i] = xs[i] = x_i;
        }

        std::sort(xd.begin(), xd.end());
        std::sort(xs.begin(), xs.end());

        int errors = 0;
        int zeros = 0;
        for (unsigned i = 0; i < xd.size(); i++) {
            errors += (xd[i] == xs[i]) ? 0 : 1;
            zeros += (xd[i] == 0.0) ? 1 : 0;
        }

        DRWN_LOG_MESSAGE("..." << errors << " errors on vector with " << zeros << " zeros");
        DRWN_ASSERT_MSG((errors == 0) && (zeros + xs.nnz() == VEC_SIZE),
            "(" << errors << " == 0) && (" << zeros << " + " << xs.nnz() << " == " << VEC_SIZE << ")");
    }
#endif

    DRWN_FCN_TOC;
}

void testMaxSpeed()
{
    const int NUM_REPEATS = 100;
    const int VEC_SIZE = 8 * 1024 * 1024;
    vector<double> x(VEC_SIZE);
    for (unsigned i = 0; i < x.size(); i++) {
         x[i] = drand48() - 0.5;
    }

    int h1 = drwnCodeProfiler::getHandle("if-then");
    int h2 = drwnCodeProfiler::getHandle("multiply");
    int h3 = drwnCodeProfiler::getHandle("std::max");
    int h4 = drwnCodeProfiler::getHandle("std::max_element");
    int h5 = drwnCodeProfiler::getHandle("inline");

    for (int r = 0; r < NUM_REPEATS; r++) {

        // if-then
        drwnCodeProfiler::tic(h1);
        double maxVal1 = -DRWN_DBL_MAX;
        for (unsigned i = 0; i < x.size(); i++) {
            if (maxVal1 < x[i]) {
                maxVal1 = x[i];
            }
        }
        drwnCodeProfiler::toc(h1);

        // multiply
        drwnCodeProfiler::tic(h2);
        double maxVal2 = -DRWN_DBL_MAX;
        for (unsigned i = 0; i < x.size(); i++) {
            maxVal2 = (maxVal2 < x[i]) * (x[i] - maxVal2) + maxVal2;
        }
        drwnCodeProfiler::toc(h2);

        // std::max
        drwnCodeProfiler::tic(h3);
        double maxVal3 = -DRWN_DBL_MAX;
        for (unsigned i = 0; i < x.size(); i++) {
            maxVal3 = std::max(maxVal3, x[i]);
        }
        drwnCodeProfiler::toc(h3);

        // std::max_element
        drwnCodeProfiler::tic(h4);
        double maxVal4 = *std::max_element(x.begin(), x.end());
        drwnCodeProfiler::toc(h4);

        // inline
        drwnCodeProfiler::tic(h5);
        double maxVal5 = -DRWN_DBL_MAX;
        for (unsigned i = 0; i < x.size(); i++) {
            maxVal5 = (maxVal5 < x[i]) ? x[i] : maxVal5;
        }
        drwnCodeProfiler::toc(h5);

        DRWN_ASSERT(maxVal2 == maxVal1);
        DRWN_ASSERT(maxVal3 == maxVal1);
        DRWN_ASSERT(maxVal4 == maxVal1);
        DRWN_ASSERT(maxVal5 == maxVal1);
    }
}

void testFastExp()
{
    const int NUM_REPEATS = 100;
    const int VEC_SIZE = 1024 * 1024;
    const double MIN_VAL = -100.0;
    const double MAX_VAL = 0.0;

    vector<double> x(VEC_SIZE);
    for (unsigned i = 0; i < x.size(); i++) {
        x[i] = MIN_VAL + (MAX_VAL - MIN_VAL) * i / x.size();
    }

    int h1 = drwnCodeProfiler::getHandle("exp");
    int h2 = drwnCodeProfiler::getHandle("fastexp");

    vector<double> y(VEC_SIZE);
    vector<double> fy(VEC_SIZE);

    DRWN_LOG_MESSAGE("testing exp() and fastexp() for " << NUM_REPEATS * VEC_SIZE << " calls...");
    for (int r = 0; r < NUM_REPEATS; r++) {

        // exp
        drwnCodeProfiler::tic(h1);
        for (unsigned i = 0; i < x.size(); i++) {
            y[i] = exp(x[i]);
        }
        drwnCodeProfiler::toc(h1);

        // fast
        drwnCodeProfiler::tic(h2);
        for (unsigned i = 0; i < x.size(); i++) {
            fy[i] = drwn::fastexp(x[i]);
        }
        drwnCodeProfiler::toc(h2);
    }

    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    for (unsigned i = 0; i < x.size(); i++) {
        maxAbsError = std::max(maxAbsError, fabs(y[i] - fy[i]));
        maxRelError = std::max(maxRelError, fabs((y[i] - fy[i]) / y[i]));
    }

    DRWN_LOG_MESSAGE("...max absolute error: " << maxAbsError);
    DRWN_LOG_MESSAGE("...max relative error: " << maxRelError);
}

void testDotProduct()
{
    const int N = 1001;
    vector<double> x(N);
    vector<double> y(N);

    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
    }

    vector<double> d(5);

    d[0] = drwn::dot(x, y);
    d[1] = drwn::dot(&x[0], &y[0], x.size());

    drwnSparseVec<double> sx(x);
    drwnSparseVec<double> sy(y);

    d[2] = drwnSparseVec<double>::dot(sx, y);
    d[3] = drwnSparseVec<double>::dot(x, sy);
    d[4] = drwnSparseVec<double>::dot(sx, sy);

    DRWN_LOG_MESSAGE("dot product: " << toString(d));
    for (unsigned i = 0; i < d.size(); i++) {
        for (unsigned j = i + 1; j < d.size(); j++) {
            DRWN_ASSERT_MSG(fabs(d[i] - d[j]) < 1.0e-6,
                "error: " << d[i] - d[j] << " (" << i << ", " << j << ")");
        }
    }
}

void testFeatureWhitener()
{
    const unsigned NUM_SAMPLES = 100000;
    const unsigned VEC_SIZE = 1000;

    // generate data
    DRWN_LOG_MESSAGE("generating data...");
    vector<vector<double> > x(NUM_SAMPLES, vector<double>(VEC_SIZE));

    srand48(0);
    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        for (unsigned j = 0; j < VEC_SIZE; j++) {
            x[i][j] = (double)j + 2.0 * (drand48() - 0.5);
            //cout << x[i][j] << (j == VEC_SIZE - 1 ? ";\n" : " ");
        }
    }

    // compute prior statistics
    DRWN_LOG_MESSAGE("computing initial statistics...");
    drwnSuffStats priorSuffStats(VEC_SIZE, DRWN_PSS_DIAG);
    priorSuffStats.accumulate(x);

    // whiten features
    DRWN_LOG_MESSAGE("whitening features...");
    drwnFeatureWhitener whitener;
    whitener.train(x);
    whitener.transform(x);

    // compute posterior statistics
    DRWN_LOG_MESSAGE("computing final statistics...");
    drwnSuffStats posteriorSuffStats(VEC_SIZE, DRWN_PSS_DIAG);
    posteriorSuffStats.accumulate(x);

    // show results
    for (unsigned i = 0; i < VEC_SIZE; i++) {
        double mu = priorSuffStats.sum(i) / priorSuffStats.count();
        double mu2 = posteriorSuffStats.sum(i) / posteriorSuffStats.count();
        double sigma = priorSuffStats.sum2(i,i) / priorSuffStats.count() - mu * mu;
        double sigma2 = posteriorSuffStats.sum2(i,i) / posteriorSuffStats.count() - mu2 * mu2;
        cout << mu << "\t" << mu2 << "\t|\t" << sigma << "\t" << sigma2 << "\n";
    }
}

void testSuffStats()
{
    double data[5][4] = { {1.0, 0.0, 0.0, 0.0},
                          {1.0, 1.0, 0.0, 0.0},
                          {1.0, 1.0, 1.0, 0.0},
                          {1.0, 1.0, 1.0, 1.0},
                          {0.5, 0.0, 0.5, 0.0} };

    vector<double> x(4);

    // full sufficient statistics
    drwnSuffStats stats(x.size(), DRWN_PSS_FULL);
    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < x.size(); j++) {
            x[j] = data[i][j];
        }
        stats.accumulate(x, (double)i);
    }

    cout << stats.count() << endl;
    cout << stats.firstMoments().transpose() << endl;
    cout << stats.secondMoments() << endl << endl;

    stats.diagonalize();
    cout << stats.count() << endl;
    cout << stats.firstMoments().transpose() << endl;
    cout << stats.secondMoments() << endl << endl;

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < x.size(); j++) {
            x[j] = data[i][j];
        }
        stats.subtract(x, (double)i);
    }

    cout << stats.count() << endl;
    cout << stats.firstMoments().transpose() << endl;
    cout << stats.secondMoments() << endl << endl;

    stats.clear();
    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < x.size(); j++) {
            x[j] = data[i][j];
        }
        stats.accumulate(x, (double)i);
    }

    cout << stats.count() << endl;
    cout << stats.firstMoments().transpose() << endl;
    cout << stats.secondMoments() << endl << endl;
}

void testPCA()
{
    VectorXd mu = VectorXd::Zero(5);
    MatrixXd sigma(5, 5);
    sigma << 0.99100, 0.91660, 1.04247, 0.75531, 0.49842,
        0.91660, 2.00318, 1.81101, 0.51745, 1.00774,
        1.04247, 1.81101, 2.09615, 0.55339, 1.36639,
        0.75531, 0.51745, 0.55339, 0.67500, 0.15996,
        0.49842, 1.00774, 1.36639, 0.15996, 1.16316;

    drwnPCA pca(drwnSuffStats(1.0, mu, sigma));

    pca.dump();

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "test", NULL, false);
    pca.save(*node);

    drwnXMLNode* child = node->first_node("translation");
    VectorXd translation;
    drwnXMLUtils::deserialize(*child, translation);
    cout << translation.transpose() << endl;

    child = node->first_node("projection");
    MatrixXd projection;
    drwnXMLUtils::deserialize(*child, projection);
    cout << projection << endl;

    sigma = projection * sigma * projection.transpose();
    drwnPCA pca2(drwnSuffStats(1.0, mu, sigma));

    xml.clear();
    node = drwnAddXMLChildNode(xml, "test", NULL, false);
    pca2.save(*node);

    child = node->first_node("translation");
    drwnXMLUtils::deserialize(*child, translation);
    cout << translation.transpose() << endl;

    child = node->first_node("projection");
    drwnXMLUtils::deserialize(*child, projection);
    cout << projection << endl;

    cout << "\n----------------------------------------------------\n\n";

    drwnPCA pca3;
    vector<vector<double> > features;
    for (unsigned i = 0; i < 1000; i++) {
        features.push_back(vector<double>(5));
        Eigen::Map<VectorXd>(&(features.back()[0]), 5) = VectorXd::Random(5);
    }
    pca3.train(features);

    xml.clear();
    node = drwnAddXMLChildNode(xml, "test", NULL, false);
    pca3.save(*node);

    child = node->first_node("translation");
    drwnXMLUtils::deserialize(*child, translation);
    cout << translation.transpose() << endl;

    child = node->first_node("projection");
    drwnXMLUtils::deserialize(*child, projection);
    cout << projection << endl;

    pca3.train(features, drwnTFeatureMapTransform<drwnIdentityFeatureMap>());

    xml.clear();
    node = drwnAddXMLChildNode(xml, "test", NULL, false);
    pca3.save(*node);

    child = node->first_node("translation");
    drwnXMLUtils::deserialize(*child, translation);
    cout << translation.transpose() << endl;

    child = node->first_node("projection");
    drwnXMLUtils::deserialize(*child, projection);
    cout << projection << endl;

    pca3.train(features, drwnTFeatureMapTransform<drwnSquareFeatureMap>());

    xml.clear();
    node = drwnAddXMLChildNode(xml, "test", NULL, false);
    pca3.save(*node);

    child = node->first_node("translation");
    drwnXMLUtils::deserialize(*child, translation);
    cout << translation.transpose() << endl;

    child = node->first_node("projection");
    drwnXMLUtils::deserialize(*child, projection);
    cout << projection << endl;
}

void testFisherLDA()
{
    VectorXd muA(2), muB(2);
    muA << 0.5, 0.5;
    muB << -0.5, -0.5;

    MatrixXd sigmaA(2, 2);
    sigmaA << 0.0964, -0.0505, -0.0505, 0.0540;
    sigmaA += muA * muA.transpose();

    MatrixXd sigmaB(2, 2);
    sigmaB << 0.1169, -0.1005, -0.1005, 0.1985;
    sigmaB += muB * muB.transpose();

    drwnCondSuffStats stats(2, 2);
    stats.accumulate(drwnSuffStats(1, muA, sigmaA), 0);
    stats.accumulate(drwnSuffStats(1, muB, sigmaB), 1);

    drwnFisherLDA lda;
    lda.train(stats);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "lda", NULL, false);

    drwnXMLEncoderType savedEncoderType = drwnXMLUtils::DEFAULT_ENCODER;
    drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;
    lda.save(*node);
    drwnXMLUtils::DEFAULT_ENCODER = savedEncoderType;
    drwnXMLUtils::dump(*node);
}

void testKMeans()
{
    const size_t N = 1000;
    const size_t D = 3;
    const unsigned K = 5;

    vector<vector<double> > x(N, vector<double>(D, 0.0));
    for (unsigned n = 0; n < N; n++) {
        for (unsigned d = 0; d < D; d++) {
            x[n][d] = drand48();
        }
    }

    drwnKMeans clustering(K);
    clustering.train(x);

    vector<vector<double> > y(N, vector<double>(K, 0.0));
    clustering.transform(x, y);

    vector<int> index = drwn::argmins(y);

    drwnXMLDoc xml;
    drwnXMLNode *node = drwnAddXMLChildNode(xml, "kmeans", NULL, false);

    drwnXMLEncoderType savedEncoderType = drwnXMLUtils::DEFAULT_ENCODER;
    drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_TEXT;
    clustering.save(*node);
    drwnXMLUtils::DEFAULT_ENCODER = savedEncoderType;
    drwnXMLUtils::dump(*node);
}

void testGaussianSampling()
{
    VectorXd mu(2);
    mu << -1.0, 1.0;
    MatrixXd sigma(2, 2);
    sigma << 1.0, 0.5, 0.5, 1.0;

    drwnGaussian g(mu, sigma);

    drwnSuffStats s(2);
    vector<double> x(2);
    for (int i = 0; i < 1000; i++) {
        g.sample(x);
        s.accumulate(x);
        cout << Eigen::Map<VectorXd>(&x[0], x.size()).transpose() << "\n";
    }

    drwnGaussian h(s);
    g.dump();
    h.dump();
}

void testGaussianMixture()
{
    VectorXd mu(2);
    MatrixXd sigma(2, 2);

    mu << -1.0, 1.0;
    sigma << 1.0, 0.5, 0.5, 1.0;
    drwnGaussian g(mu, sigma);

    mu << 1.0, -1.0;
    sigma << 1.0, -0.5, -0.5, 1.0;
    drwnGaussian h(mu, sigma);

    vector<vector<double> > x(5000, vector<double>(2));
    for (unsigned i = 0; i < x.size(); i += 2) {
        g.sample(x[i]);
        h.sample(x[i + 1]);
    }

    drwnGaussianMixture model(2, 3);
    model.train(x, 1.0e-6);

    model.dump();
}

void testPRCurve()
{
    const int N = 100;
    vector<int> y(N);
    vector<double> y_hat(N);

    for (int n = 0; n < N; n++) {
        y[n] = n % 2;
        y_hat[n] = y[n] + 2.0 * drand48() - 1.0;
    }

    // compute confusion matrix
    drwnConfusionMatrix confusion(2);
    for (int n = 0; n < N; n++) {
        confusion.accumulate(y[n], y_hat[n] > 0.5);
    }
    double p = confusion.precision(1);
    double r = confusion.recall(1);

    // gerenate full pr curve
    drwnPRCurve curve;
    for (int n = 0; n < N; n++) {
        if (y[n] == 1) curve.accumulatePositives(y_hat[n]);
        else curve.accumulateNegatives(y_hat[n]);
    }

    bool bFound = false;
    vector<pair<double, double> > points = curve.getCurve();
    for (vector<pair<double, double> >::const_iterator it = points.begin();
         it != points.end(); ++it) {
        cout << it->first << " " << it->second << "\n";
        bFound = bFound || ((it->first == p) && (it->second == r));
    }

    DRWN_ASSERT_MSG(bFound, "testPRCurve() failed");
}

void testCrossValidator()
{
    drwnCrossValidator cv;
    cv.add("regularizer", "0"); // L2
    cv.add("regularizer", "1"); // huber
    cv.addLogarithmic("regStrength", 1.0e-3, 1.0e3, 7);
    cv.dump();

    drwnClassifierDataset trainSet;
    drwnClassifierDataset testSet;
    for (int i = 0; i < 1000; i++) {
        trainSet.append(vector<double>(1, (i % 2) + 2.0 * drand48()), i % 2);
        testSet.append(vector<double>(1, (i % 2) + 2.0 * drand48()), i % 2);
    }

    drwnClassifier *classifier =
        new drwnMultiClassLogistic(trainSet.numFeatures(), trainSet.maxTarget() + 1);

    cv.crossValidate(classifier, trainSet, testSet);

    drwnConfusionMatrix confusion(classifier->numClasses());
    confusion.accumulate(testSet, classifier);
    confusion.printCounts();

    delete classifier;
}

void testQPSolver(const char *filename, bool bLogBarrier)
{
    DRWN_ASSERT(filename != NULL);

    // read problem from XML file
    drwnXMLDoc xml;
    drwnXMLNode *node = drwnParseXMLFile(xml, filename, "qp");

    // read P, q and r
    MatrixXd P;
    drwnXMLNode *subnode = node->first_node("P");
    drwnXMLUtils::deserialize(*subnode, P);

    VectorXd q;
    subnode = node->first_node("q");
    if (subnode == NULL) {
        q = VectorXd::Zero(P.rows());
    } else {
        drwnXMLUtils::deserialize(*subnode, q);
    }

    double r = 0.0;
    subnode = node->first_node("r");
    if (subnode != NULL) {
        r = atof(drwnGetXMLText(*subnode));
    }

    // create QP solver
    drwnQPSolver *solver = NULL;
    if (bLogBarrier) {
        solver = new drwnLogBarrierQPSolver(P, q, r);
    } else {
        solver = new drwnQPSolver(P, q, r);
    }

    // read equality constraints
    subnode = node->first_node("A");
    if (subnode != NULL) {
        MatrixXd A;
        drwnXMLUtils::deserialize(*subnode, A);
        VectorXd b;
        subnode = node->first_node("b");
        DRWN_ASSERT(subnode != NULL);
        drwnXMLUtils::deserialize(*subnode, b);

        // set the constraints
        solver->setEqConstraints(A, b);
    }

    // read inequality constraints
    subnode = node->first_node("G");
    if (subnode != NULL) {
        MatrixXd G;
        drwnXMLUtils::deserialize(*subnode, G);
        VectorXd h;
        subnode = node->first_node("h");
        DRWN_ASSERT(subnode != NULL);
        drwnXMLUtils::deserialize(*subnode, h);

        // set the constraints
        solver->setIneqConstraints(G, h);
    }

    // read upper and lower bounds
    VectorXd lb, ub;
    subnode = node->first_node("lb");
    if (subnode != NULL) {
        drwnXMLUtils::deserialize(*node, lb);
    }
    subnode = node->first_node("ub");
    if (subnode != NULL) {
        drwnXMLUtils::deserialize(*subnode, ub);
        if (lb.rows() == 0) {
            lb = VectorXd::Constant(lb.rows(), -DRWN_DBL_MAX);
        }
    } else if (lb.rows() != 0) {
        ub = VectorXd::Constant(lb.rows(), DRWN_DBL_MAX);
    }
    if (lb.rows() != 0) {
        solver->setBounds(lb, ub);
    }

    // read x_star
    VectorXd x_star;
    subnode = node->first_node("x_star");
    if (subnode != NULL) {
        drwnXMLUtils::deserialize(*subnode, x_star);
        if (bLogBarrier) {
            ((drwnLogBarrierQPSolver *)solver)->initialize(1.0e-6 * x_star);
        }
    }

    // solve the problem
    double J = solver->solve();

    if (lb.rows() != 0) { DRWN_LOG_DEBUG("lbound is   " << toString(lb.transpose())); }
    DRWN_LOG_VERBOSE("solution is " << toString(solver->solution().transpose()));
    if (ub.rows() != 0) { DRWN_LOG_DEBUG("ubound is   " << toString(ub.transpose())); }
    DRWN_LOG_MESSAGE("objective is " << J);
    if (x_star.rows() != 0) {
        DRWN_LOG_MESSAGE("||x - x*|| = " << (solver->solution() - x_star).norm());
        DRWN_LOG_MESSAGE("    p - p* = " << J - solver->objective(x_star));
        DRWN_LOG_MESSAGE("        p* = " << solver->objective(x_star));
    }

    // free memory
    delete solver;
}

void testLPSolver()
{
    // define LP
    const int n = 100;
    const int m = 10;

    MatrixXd A = MatrixXd::Random(m, n);
    VectorXd x_feasible = VectorXd::Random(n) + VectorXd::Constant(n, 2.0);
    VectorXd b = A * x_feasible;
    VectorXd c = VectorXd::Random(n).cwiseAbs();

    // solve LP
    DRWN_LOG_MESSAGE("testing LP solver...");
    drwnLPSolver solver(c, A, b);
    double p = solver.solve();
    VectorXd x = solver.solution();
    DRWN_LOG_VERBOSE("optimal value is " << p << " at " << x.transpose());
    DRWN_LOG_VERBOSE("residual is " << (A * x - b).norm());

    // solve sparse LP
    SparseMatrix<double> sparseA(m, n);
    vector<Eigen::Triplet<double> > tripletList;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tripletList.push_back(Eigen::Triplet<double>(i, j, A(i, j)));
        }
    }
    sparseA.setFromTriplets(tripletList.begin(), tripletList.end());

    DRWN_LOG_MESSAGE("testing sparse LP solver...");
    drwnSparseLPSolver sparseSolver(c, sparseA, b);
    double p_sp = sparseSolver.solve();
    VectorXd x_sp = sparseSolver.solution();
    DRWN_LOG_VERBOSE("optimal value is " << p_sp << " at " << x_sp.transpose());
    DRWN_LOG_VERBOSE("residual is " << (A * x_sp - b).norm());
}

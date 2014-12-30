/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLinearTransform.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Jiecheng Zhao <u5143437@anu.edu.au>
**
*****************************************************************************/

#include <vector>
#include <limits>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnLinearTransform.h"

using namespace std;
using namespace Eigen;

// drwnLinearTransform class --------------------------------------------------

drwnLinearTransform::drwnLinearTransform() : drwnFeatureTransform()
{
    // do nothing
}

drwnLinearTransform::drwnLinearTransform(const VectorXd& mu, const MatrixXd& projection) :
    drwnFeatureTransform()
{
    this->set(mu, projection);
}

drwnLinearTransform::drwnLinearTransform(const drwnLinearTransform& xform) :
    drwnFeatureTransform(xform),  _mu(xform._mu), _projection(xform._projection)
{
    // do nothing
}

drwnLinearTransform::~drwnLinearTransform()
{
    // do nothing
}

// access
void drwnLinearTransform::set(const VectorXd& mu, const MatrixXd& projection)
{
    DRWN_ASSERT(mu.rows() == projection.cols());
    _mu = mu;
    _projection = projection;
    _nFeatures = _mu.rows();
    _bValid = true;
}

void drwnLinearTransform::set(const MatrixXd& projection)
{
    _mu = VectorXd::Zero(_nFeatures = _projection.cols());
    _projection = projection;
    _bValid = true;
}

// i/o
void drwnLinearTransform::clear()
{
    drwnFeatureTransform::clear();
    _mu = VectorXd::Zero(0);
    _projection = MatrixXd::Zero(0, 0);
}

bool drwnLinearTransform::save(drwnXMLNode& node) const
{
    drwnFeatureTransform::save(node);

    drwnXMLNode *child = drwnAddXMLChildNode(node, "translation", NULL, false);
    drwnXMLUtils::serialize(*child, _mu);

    child = drwnAddXMLChildNode(node, "projection", NULL, false);
    drwnXMLUtils::serialize(*child, _projection);

    return true;
}

bool drwnLinearTransform::load(drwnXMLNode& node)
{
    drwnFeatureTransform::load(node);

    drwnXMLNode *child = node.first_node("translation");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _mu);

    child = node.first_node("projection");
    DRWN_ASSERT(child != NULL);
    drwnXMLUtils::deserialize(*child, _projection);

    return true;
}

void drwnLinearTransform::transform(const vector<double>& x, vector<double>& y) const
{
    DRWN_ASSERT_MSG((int)x.size() == _nFeatures, x.size() << "!=" << _nFeatures);

    y.resize(_projection.rows());
    Eigen::Map<VectorXd>(&y[0], y.size()) = _projection *
        (Eigen::Map<const VectorXd>(&x[0], x.size()) - _mu);
}

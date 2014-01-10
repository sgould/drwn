/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnMatlabUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Utility functions for converting between Matlab and Darwin datastructures.
**
*****************************************************************************/

#pragma once

// Eigen matrix library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"
#include "drwnEngine.h"

// Matlab library
#include "mat.h"

Eigen::MatrixXd mat2eigen(const mxArray *m);
mxArray *eigen2mat(const Eigen::MatrixXd& m);

string mat2record(const mxArray *m, drwnDataRecord *r);
mxArray *record2mat(const drwnDataRecord *r, const string& key);

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnIOUtils.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

/*!
** \file drwnIOUtils.h
** \anchor drwnIOUtils
** \brief Generic i/o utilities.
*/

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "Eigen/Core"

using namespace std;
using namespace Eigen;

// matrix i/o ---------------------------------------------------------------

//! Read an Eigen::Matrix from an input stream (the \p filename argument is
//! only used for reporting errors and can be left \p NULL).
template <class EigenMatrix>
void drwnReadMatrix(EigenMatrix& m, istream& is, const char *filename = NULL)
{
    for (int y = 0; y < m.rows(); y++) {
        for (int x = 0; x < m.cols(); x++) {
            is >> m(y, x);
            DRWN_ASSERT_MSG(!is.fail(), "row/col " << y << "/" << x
                << " of " << m.rows() << "/" << m.cols() << " ("
                << (filename != NULL ? filename : "unknown file") << ")");
        }
    }
}

//! Read an Eigen::Matrix of known size from a text file.
template <class EigenMatrix>
void drwnReadMatrix(EigenMatrix& m, const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    ifstream ifs(filename);
    DRWN_ASSERT_MSG(!ifs.fail(), filename);
    drwnReadMatrix(m, ifs, filename);
    ifs.close();
}

//! Read an Eigen::Matrix of unknown size from a text file.
template <class EigenMatrix>
void drwnReadUnknownMatrix(EigenMatrix& m, const char *filename)
{
    DRWN_ASSERT(filename != NULL);
    const int nRows = drwnCountLines(filename);

    ifstream ifs(filename);
    DRWN_ASSERT_MSG(!ifs.fail(), filename);

    const int nCols = drwnCountFields(&ifs);
    m.resize(nRows, nCols);
    drwnReadMatrix(m, ifs, filename);
    ifs.close();
}


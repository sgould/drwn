/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnConstants.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

/*!
** \file drwnConstants.h
** \anchor drwnConstants
** \brief Provides useful constants and version information.
*/

#pragma once

#define DRWN_VERSION   "1.10 (beta)"
#define DRWN_TITLE     "DARWIN: A Framework for Machine Learning R&D"
#define DRWN_COPYRIGHT "Copyright (c) 2007-2017, Stephen Gould"

#define DRWN_USAGE_HEADER (DRWN_TITLE " (Version: " DRWN_VERSION ")\n" DRWN_COPYRIGHT "\n")

#define DRWN_EPSILON (1.0e-6)
#define DRWN_DBL_MIN numeric_limits<double>::min()
#define DRWN_DBL_MAX numeric_limits<double>::max()
#define DRWN_FLT_MIN numeric_limits<float>::min()
#define DRWN_FLT_MAX numeric_limits<float>::max()
#define DRWN_INT_MIN numeric_limits<int>::min()
#define DRWN_INT_MAX numeric_limits<int>::max()

#define DRWN_NOT_IMPLEMENTED_YET { \
    std::cerr << "ERROR: function \"" << __FUNCTION__ << "\" not implemented yet" << std::endl; \
    assert(false); \
}

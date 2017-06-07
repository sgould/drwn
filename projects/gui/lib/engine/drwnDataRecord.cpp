/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDataRecord.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>
#include <map>
#include <vector>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnDataRecord.h"
#include "drwnDatabase.h"

using namespace std;
using namespace Eigen;

// drwnDataRecord ------------------------------------------------------------

drwnDataRecord::drwnDataRecord(drwnDataTable *o) :
    _owner(o), _bDirty(false)
{
    // do nothing
}

drwnDataRecord::drwnDataRecord(const drwnDataRecord& r) :
    _owner(r._owner), _bDirty(r._bDirty), _structure(r._structure),
    _data(r._data), _objective(r._objective), _gradient(r._gradient)
{
    // do nothing
}

drwnDataRecord::~drwnDataRecord()
{
    // do nothing
}

void drwnDataRecord::clear()
{
    if (isEmpty()) return;

    _bDirty = true;
    _structure = VectorXi::Zero(0);
    _data = MatrixXd::Zero(0,0);
    _objective = VectorXd::Zero(0);
    _gradient = MatrixXd::Zero(0,0);
}

bool drwnDataRecord::read(istream& is)
{
    int rows = 0;
    int cols = 0;

    is.read((char *)&rows, sizeof(int));
    DRWN_ASSERT_MSG(!is.fail() && (rows >= 0), 
        (_owner ? _owner->name().c_str() : "<unknown>"));

    _structure.resize(rows);

    if (rows > 0) {
        is.read((char *)_structure.data(), rows * sizeof(int));
    }
    DRWN_ASSERT(!is.fail());

    is.read((char *)&rows, sizeof(int));
    is.read((char *)&cols, sizeof(int));
    DRWN_ASSERT(!is.fail() && (rows >= 0) && (cols >= 0));

    _data.resize(rows, cols);

    if (rows > 0) {
        is.read((char *)_data.data(), rows * cols * sizeof(double));
    }
    DRWN_ASSERT(!is.fail());

    is.read((char *)&rows, sizeof(int));
    DRWN_ASSERT(!is.fail() && (rows >= 0));

    _objective.resize(rows);
    if (rows > 0) {
        is.read((char *)_objective.data(), rows * sizeof(double));
    }

    is.read((char *)&rows, sizeof(int));
    is.read((char *)&cols, sizeof(int));
    DRWN_ASSERT(!is.fail() && (rows >= 0) && (cols >= 0));

    _gradient.resize(rows, cols);
    if (rows > 0) {
        is.read((char *)_gradient.data(), rows * cols * sizeof(double));
    }
    DRWN_ASSERT(!is.fail());

    _bDirty = false;
    return true;
}

bool drwnDataRecord::write(ostream& os) const
{
    // TODO: fix 32-bit/64-bit problems
    int rows = _structure.rows();
    os.write((char *)&rows, sizeof(int));
    if (rows > 0) {
        os.write((char *)_structure.data(), rows * sizeof(int));
    }

    rows = _data.rows();
    int cols = _data.cols();
    os.write((char *)&rows, sizeof(int));
    os.write((char *)&cols, sizeof(int));
    if (rows > 0) {
        os.write((char *)_data.data(), rows * cols * sizeof(double));
    }

    rows = _objective.rows();
    os.write((char *)&rows, sizeof(int));
    if (rows > 0) {
        os.write((char *)_objective.data(), rows * sizeof(double));
    }

    rows = _gradient.rows();
    cols = _gradient.cols();
    os.write((char *)&rows, sizeof(int));
    os.write((char *)&cols, sizeof(int));
    if (rows > 0) {
        os.write((char *)_gradient.data(), rows * cols * sizeof(double));
    }

    DRWN_ASSERT_MSG(!os.fail(), "bytes: " << this->numBytesOnDisk());
    _bDirty = false;
    return true;
}

void drwnDataRecord::swap(drwnDataRecord& r)
{
    std::swap(_owner, r._owner);
    std::swap(_bDirty, r._bDirty);
    std::swap(_structure, r._structure);
    std::swap(_data, r._data);
    std::swap(_objective, r._objective);
    std::swap(_gradient, r._gradient);
}

drwnDataRecord& drwnDataRecord::operator=(drwnDataRecord& r)
{
    this->swap(r);
    return *this;
}

bool drwnDataRecord::operator==(const drwnDataRecord& r) const
{
    return ((r._structure == _structure) && (r._data == _data) && 
        (r._objective == _objective) && (r._gradient == _gradient));
}

bool drwnDataRecord::operator!=(const drwnDataRecord& r) const
{
    return !(*this == r);
}

bool drwnDataRecord::checkDataSizes() const
{
    if (hasStructure() && hasData()) {
        int dim = _structure[0];
        for (int i = 1; i < _structure.size(); i++) {
            dim *= _structure[i];
        }
        DRWN_ASSERT(_data.rows() == dim);
    }

    if (hasData() && hasGradient()) {
        DRWN_ASSERT(_data.rows() == _gradient.rows());
        DRWN_ASSERT(_data.cols() == _gradient.cols());
    }

    if (hasData() && hasObjective()) {
        DRWN_ASSERT(_data.rows() == _objective.rows());
    }

    if (hasGradient() && hasObjective()) {
        DRWN_ASSERT(_gradient.rows() == _objective.rows());
    }

    return true;
}

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDataRecord.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Defines record datastructure that is manipulated by processing nodes. See
**  also drwnDataTable, drwnDatabase and associated classes.
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <map>
#include <list>
#include <vector>

#include "Eigen/Core"

#include "drwnBase.h"
#include "drwnIO.h"

using namespace std;
using namespace Eigen;

// forward declarations ------------------------------------------------------

class drwnDatabase;
class drwnDataTable;

// drwnDataRecord ------------------------------------------------------------
// Encapsulates the data, objective and gradient of a single training or test
// instance. Meta-data (such as image size, etc) is encoded through the
// structure member variable.

class drwnDataRecord : public drwnPersistentRecord {
 protected:
    drwnDataTable *_owner;   // table that owns this record
    mutable bool _bDirty;    // true if data has been modified (different from disk)

    VectorXi _structure;     // data structure (e.g. matrix dimensions)
    MatrixXd _data;          // forward data
    VectorXd _objective;     // backward data
    MatrixXd _gradient;

 public:
    drwnDataRecord(drwnDataTable *o = NULL);
    drwnDataRecord(const drwnDataRecord& r);
    virtual ~drwnDataRecord();

    drwnDataTable *getOwner() const { return _owner; }
    void setOwner(drwnDataTable *o) { _owner = o; }

    int numObservations() const { return _data.rows(); }
    int numFeatures() const { return _data.cols(); }
    int numBytes() const { return (_structure.size() * sizeof(int) +
        (_data.size() + _objective.size() + _gradient.size()) * sizeof(double)); }
    size_t numBytesOnDisk() const { return (size_t)(numBytes() + 6 * sizeof(int)); }

    bool isDirty() const { return _bDirty; }
    bool hasStructure() const { return _structure.size() != 0; }
    bool hasData() const { return _data.size() != 0; }
    bool hasObjective() const { return _objective.size() != 0; }
    bool hasGradient() const { return _gradient.size() != 0; }
    bool isEmpty() const { return !hasStructure() && !hasData() && !hasObjective() && !hasGradient(); }

    void clear();
    bool read(istream& is);
    bool write(ostream& os) const;
    void swap(drwnDataRecord& r);

    // data access
    const VectorXi &structure() const { return _structure; }
    const MatrixXd &data() const { return _data; }
    const VectorXd &objective() const { return _objective; }
    const MatrixXd &gradient() const { return _gradient; }

    // data modification
    VectorXi &structure() { _bDirty = true; return _structure; }
    MatrixXd &data() { _bDirty = true; return _data; }
    VectorXd &objective() { _bDirty = true; return _objective; }
    MatrixXd &gradient() { _bDirty = true; return _gradient; }

    drwnDataRecord& operator=(drwnDataRecord& r);
    bool operator==(const drwnDataRecord& r) const;
    bool operator!=(const drwnDataRecord& r) const;

 protected:
    bool checkDataSizes() const;
};

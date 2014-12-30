/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDisjointSets.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>

// drwnDisjointSets ---------------------------------------------------------
//! Implements a forest of disjoint sets abstract data type.
//!
//! The elements are numbered from 0 to (size() - 1). Each element
//! belongs to exactly one set. The sets have ID sparesly numbered in the
//! range 0 to (size() - 1). To get the number of elements of each set,
//! call size(id) with a valid set ID.
//!
//! \sa \ref drwnDisjointSetsDoc

class drwnDisjointSets {
 protected:
    typedef struct _node_t {
        int rank;                   // depth of node
        int size;                   // size of set (for root nodes)
        int index;                  // index of the element represented by this node
        _node_t *parent;            // parent of this node
    } node_t;

    int _nElements;                 //!< number of elements in the forest
    int _nSets;                     //!< number of sets in the forest
    std::vector<node_t> _nodes;     //!< list of nodes

 public:
    //! construct a disjoint dataset with \p count nodes
    drwnDisjointSets(unsigned int count = 0);
    //! copy constructor
    drwnDisjointSets(const drwnDisjointSets &dset);
    //! destructor
    ~drwnDisjointSets();

    //! return the total number of nodes in the disjoint set
    inline int size() const { return _nElements; }
    //! return the number of nodes in set \p setId
    int size(int setId) const;
    //! return the number of disjoint sets
    inline int sets() const { return _nSets; }
    //! get the ids of nodes in set with id \p setId
    std::set<int> getMembers(int setId);
    //! add \p count (disconnected) nodes
    void add(int count);
    //! find which set a particular node belongs to
    int find(int elementId);
    //! join two sets and return the joined set id (either \p setId1 or \p setId2)
    int join(int setId1, int setId2);
    //! return a list of valid set ids
    std::vector<int> getSetIds() const;

    //! assignment operator
    drwnDisjointSets& operator=(const drwnDisjointSets& dset);
};

/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2015, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnDisjointSets.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>

#include "drwnBase.h"
#include "drwnDisjointSets.h"

using namespace std;

// drwnDisjointSets ---------------------------------------------------------

drwnDisjointSets::drwnDisjointSets(unsigned int count) :
    _nElements(0), _nSets(0)
{
    add(count);
}

drwnDisjointSets::drwnDisjointSets(const drwnDisjointSets &dset) :
    _nElements(dset._nElements), _nSets(dset._nSets)
{
    // copy nodes
    _nodes.resize(_nElements);
    for (int i = 0; i < _nElements; i++) {
        _nodes[i] = dset._nodes[i];
    }

    // update parent pointers
    for (int i = 0; i < _nElements; i++) {
        if (_nodes[i].parent != NULL) {
            _nodes[i].parent = &_nodes[dset._nodes[i].parent->index];
        }
    }
}

drwnDisjointSets::~drwnDisjointSets()
{
    //_nodes.clear();
}

int drwnDisjointSets::size(int setId) const
{
    DRWN_ASSERT(setId < _nElements);
    DRWN_ASSERT(_nodes[setId].parent == NULL);

    return _nodes[setId].size;
}

set<int> drwnDisjointSets::getMembers(int setId)
{
    set<int> members;
    for (int i = 0; i < _nElements; i++) {
        if (find(i) == setId)
            members.insert(i);
    }

    return members;
}

void drwnDisjointSets::add(int count)
{
    _nodes.resize(_nElements + count);
    for (int i = 0; i < count; i++) {
        _nodes[_nElements + i].index = _nElements + i;
        _nodes[_nElements + i].size = 1;
        _nodes[_nElements + i].rank = 0;
        _nodes[_nElements + i].parent = NULL;
    }

    _nElements += count;
    _nSets += count;
}

int drwnDisjointSets::find(int elementId)
{
    DRWN_ASSERT(elementId < _nElements);

    node_t *root;
    node_t *ptrNode, *nextNode;

    // find the root node for elementId
    root = &_nodes[elementId];
    while (root->parent != NULL) {
        root = root->parent;
    }

#if 1
    // optimize for future find() operations by making all descendents of
    // the root node direct children of the root node
    ptrNode = &_nodes[elementId];
    while (ptrNode != root) {
        nextNode = ptrNode->parent;
        ptrNode->parent = root;
        ptrNode = nextNode;
    }
#else
    // optimize for future find() operations by making search node a direct
    // child of the root node
    if (&_nodes[elementId] != root) {
        _nodes[elementId].parent = root;
    }
#endif

    return (root->index);
}

int drwnDisjointSets::join(int setId1, int setId2)
{
    if (setId1 == setId2)
        return setId1;

    DRWN_ASSERT((setId1 < _nElements) && (setId2 < _nElements));
    node_t *set1, *set2;

    set1 = &_nodes[setId1];
    set2 = &_nodes[setId2];
    DRWN_ASSERT((set1->parent == NULL) && (set2->parent == NULL));

    // balance tree by joining based on rank
    int joinedSetId;
    if (set1->rank > set2->rank) {
        set2->parent = set1;
        set1->size += set2->size;
        joinedSetId = setId1;
    } else {
        set1->parent = set2;
        if (set1->rank == set2->rank) {
            set2->rank += 1;
        }
        set2->size += set1->size;
        joinedSetId = setId2;
    }

    _nSets -= 1;
    return joinedSetId;
}

vector<int> drwnDisjointSets::getSetIds() const
{
    vector<int> setIds;
    setIds.reserve(_nSets);

    for (int i = 0; i < _nElements; i++) {
        if (_nodes[i].parent == NULL) {
            setIds.push_back(i);
        }
    }

    return setIds;
}

drwnDisjointSets& drwnDisjointSets::operator=(const drwnDisjointSets& dset)
{
    _nElements = dset._nElements;
    _nSets = dset._nSets;

    // copy nodes
    _nodes = dset._nodes;

    // update parent pointers
    for (int i = 0; i < _nElements; i++) {
        if (_nodes[i].parent != NULL) {
            _nodes[i].parent = &_nodes[dset._nodes[i].parent->index];
        }
    }

    return *this;
}

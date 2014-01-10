/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2014, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnTriplet.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <ctime>
#include <vector>
#include <map>
#include <string>
#include <iostream>

// drwnTriplet class --------------------------------------------------------
//! Basic datatype for holding three objects of arbitrary type. Similar to the
//! STL pair<> class.
//!
//! \code
//! drwnTriplet<int> tuple;
//! tuple.first = 1;
//! tuple.second = 2;
//! tuple.third = 3;
//! \endcode

template<class T, class U = T, class V = T>
class drwnTriplet {
 public:
    T first;  //!< first object in the triplet
    U second; //!< second object in the triplet
    V third;  //!< third object in thr triplet

 public:
    inline drwnTriplet() { }
    inline drwnTriplet(const T& i, const U& j, const V& k) :
        first(i), second(j), third(k) { }
    inline drwnTriplet(const drwnTriplet<T,U,V>& t) :
        first(t.first), second(t.second), third(t.third) { }
    inline ~drwnTriplet() { }

    // operators
    inline drwnTriplet<T,U,V>& operator=(const drwnTriplet<T,U,V>& t);
    inline bool operator==(const drwnTriplet<T,U,V>& t) const;
    inline bool operator<(const drwnTriplet<T,U,V>& t) const;
};

// implementation -----------------------------------------------------------

template<class T, class U, class V>
inline drwnTriplet<T,U,V>& drwnTriplet<T,U,V>::operator=(const drwnTriplet<T,U,V>& t) {
    first = t.first;
    second = t.second;
    third = t.third;

    return *this;
}

template<class T, class U, class V>
inline bool drwnTriplet<T,U,V>::operator==(const drwnTriplet<T,U,V>& t) const {
    return ((t.first == first) && (t.second == second) && (t.third == third));
}

template<class T, class U, class V>
inline bool drwnTriplet<T,U,V>::operator<(const drwnTriplet<T,U,V>& t) const {
    return ((first < t.first) || (second < t.second) || (third < t.third));
}

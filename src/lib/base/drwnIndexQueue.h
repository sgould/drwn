/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnIndexQueue.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <vector>

using namespace std;

// drwnIndexQueue -----------------------------------------------------------
//! Provides a queue datastructure on a fixed number of indexes. At most one
//! copy of each index can appear in the queue (a second enqueue is ignored).
//! Membership of the queue can be queried.
//!
//! Example usage:
//! \code
//!     drwnIndexQueue q(100);
//!
//!     q.push_back(10);
//!     q.push_back(20);
//!     q.push_back(10);
//!     cout << q.front() << " is the head of the queue";
//!     q.push_front(30);
//!     cout << q.front() << " is the head of the queue";
//!     q.erase(10);
//!
//! \endcode

class drwnIndexQueue {
 protected:
    static const int TERMINAL = -1;

    int _head, _tail;
    vector<int> _rev;
    vector<int> _fwd;

 public:
    //! default constructor of an empty queue
    drwnIndexQueue() : _head(TERMINAL), _tail(TERMINAL) { /* do nothing */ }
    //! constructor takes the maximum number of indexes
    drwnIndexQueue(size_t n) : _head(TERMINAL), _tail(TERMINAL),
        _rev(n, TERMINAL), _fwd(n, TERMINAL) { /* do nothing */ }
    //! destructor
    ~drwnIndexQueue() { /* do nothing */ }

    //! returns true if the queue is empty
    inline bool empty() const { return (_head == TERMINAL); }

    //! clears the queue
    inline void clear() {
        _head = _tail = TERMINAL;
        fill(_rev.begin(), _rev.end(), TERMINAL);
        fill(_fwd.begin(), _fwd.end(), TERMINAL);
    }

    //! clears and resizes the queue
    inline void resize(size_t n) {
        _rev.resize(n); _fwd.resize(n); clear();
    }

    //! return the element at the front of the queue (or -1 if empty)
    inline int front() const { return _head; }
    //! return the element at the back of the queue (or -1 if empty)
    inline int back() const { return _tail; }

    //! returns true if \p u is in the queue
    inline bool is_queued(int u) const {
        return (_rev[u] != TERMINAL);
    }

    //! add \p u to the back of the queue
    inline void push_back(int u) {
        if (is_queued(u)) return;

        if (_tail == TERMINAL) {
            _head = _tail = _fwd[u] = _rev[u] = u;
        } else {
            _fwd[u] = _head;
            _rev[u] = _tail;
            _fwd[_tail] = u;
            _rev[_head] = u;
            _tail = u;
        }
    }

    //! add \p u to the front of the queue
    inline void push_front(int u) {
        if (is_queued(u)) return;

        if (_head == TERMINAL) {
            _head = _tail = _fwd[u] = _rev[u] = u;
        } else {
            _fwd[u] = _head;
            _rev[u] = _tail;
            _fwd[_tail] = u;
            _rev[_head] = u;
            _head = u;
        }
    }

    //! remove the elelemt from the head of the queue
    inline void pop_front() {
        DRWN_ASSERT(_head != TERMINAL);
        _rev[_head] = TERMINAL;
        if (_head == _tail) {
            _head = _tail = TERMINAL;
        } else {
            _head = _fwd[_head];
            _rev[_head] = _tail;
            _fwd[_tail] = _head;
        }
    }

    //! remove \p u from the queue or do nothing if not in the queue
    inline void erase(int u) {
        if (!is_queued(u)) return;

        if (_head == _tail) {
            _head = _tail = TERMINAL;
        } else {
            _fwd[_rev[u]] = _fwd[u];
            _rev[_fwd[u]] = _rev[u];
            if (_head == u) _head = _fwd[u];
            if (_tail == u) _tail = _rev[u];
        }

        _rev[u] = TERMINAL;
    }
};

//! specialized toString() routine
std::string toString(const drwnIndexQueue& q);

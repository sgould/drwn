/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLPSolver.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "Eigen/Core"
#include "Eigen/SparseCore"

#include "drwnBase.h"

using namespace std;
using namespace Eigen;

// drwnLPSolver ------------------------------------------------------------

/*!
** \brief Solves equality constrained linear programs with positivity
** constraints via the log-barrier method.
**
** Solves linear programs of the form:
** \f[
**    \begin{array}{ll}
**    \textrm{minimize (over $x$)} & c^T x \\
**    \textrm{subject to}          & Ax = b \\
**                                 & x \geq 0
**    \end{array}
** \f]
**
** or, more generally,
**
** \f[
**    \begin{array}{ll}
**    \textrm{minimize (over $x$)} & c^T x \\
**    \textrm{subject to}          & Ax = b \\
**                                 & l \leq x \leq u \\
**    \end{array}
** \f]
**
** General inequality constraints of the form \f$g^T x \leq h\f$ can
** be handled by introducing slack variables, e.g., \f$g^T x + z = h\f$
** with \f$z \geq 0\f$.
**
** The following code snippet shows an example of solving the problem
** \f$min. \sum_i x_i \text{ s.t. } x_1 + x_2 = 1, x_2 + x_3 = 1, x_i \geq 0\f$.
**
** \code
**     VectorXd b(2), c(3);
**     MatrixXd A(2, 3);
**
**     c << 1, 1, 1;
**     A << 1, 1, 0, 0, 1, 1;
**     b << 1, 1;
**
**     drwnLPSolver solver(c, A, b);
**     double p_star = solver.solve();
**     DRWN_LOG_MESSAGE("optimal value is " << p_star);
**
**     VectorXd x_star = solver.solution();
**     DRWN_LOG_MESSAGE("solution is " << x_star.transpose());
** \endcode
**
** See:
**   \li S. Boyd and L. Vandenberghe, Convex Optimization, 2004.
*/

class drwnLPSolver {
 public:
    // barrier method parameters
    static double t0;         //!< initial barrier function multiplier
    static double mu;         //!< barrier function multiplier update
    static double eps;        //!< stopping tolerance
    static unsigned maxiters; //!< maximum number of newton steps per iteration
    // line search parameters
    static double alpha;      //!< line search stopping criterion in (0, 1/2)
    static double beta;       //!< line search backtracking parameter in (0, 1)

 protected:
    // objective
    VectorXd _c;              //!< linear term in the objective function

    // constraints and bounds
    MatrixXd _A;              //!< linear equality constraint matrix
    VectorXd _b;              //!< linear equality constraint vector

    VectorXd _lb;             //!< lower bound for each variable (-DRWN_DBL_MAX for unbounded below)
    VectorXd _ub;             //!< upped bound for each variable (DRWN_DBL_MAX for unbounded above)

    VectorXd _x;              //!< current estimate of solution

 public:
    //! construct a problem \f$min. c^T x \text{ s.t. } Ax = b, x \geq 0\f$
    drwnLPSolver(const VectorXd& c, const MatrixXd& A, const VectorXd& b);
    //! construct a problem \f$min. c^T x \text{ s.t. } Ax = b, l \leq x \leq u\f$
    drwnLPSolver(const VectorXd& c, const MatrixXd& A, const VectorXd& b,
        const VectorXd& lb, const VectorXd& ub);
    //! destructor
    ~drwnLPSolver() { /* do nothing */ }

    //! initialization of a feasible point
    void initialize(const VectorXd& x);

    //! solve and return objective (solution can be obtained from \ref solution function)
    double solve();

    //! return the current estimate of the solution
    inline const VectorXd& solution() const { return _x; }

    // accessors
    //! return the number of dimensions of the state space
    inline int size() const { return _c.rows(); }    
    //! access the \p i-th dimension of the current solution
    inline double operator[](unsigned i) const { return _x[i]; }

 protected:
    //! returns true if the i-th variable is unbounded
    bool isUnbounded(unsigned i) const { return ((_lb[i] == -DRWN_DBL_MAX) && (_ub[i] == DRWN_DBL_MAX)); }
    //! returns true if a point is strictly within the lower and upper bounds
    bool isWithinBounds(const VectorXd& x) const { return (x.array() > _lb.array()).all() && (x.array() < _ub.array()).all(); }

    //! computes the barrier function for a given assignment
    double barrierFunction(const VectorXd& x) const;
};

// drwnSparseLPSolver ------------------------------------------------------

/*!
** \brief Solves linear programs with sparse equality constraints.
**
** \todo refactor drwnLPSolver code to use templated sparse/dense matrix
**
** \sa drwnLPSolver
*/

class drwnSparseLPSolver {
 protected:
    // objective
    VectorXd _c;              //!< linear term in the objective function

    // constraints and bounds
    const SparseMatrix<double>& _A;  //!< linear equality constraint matrix
    VectorXd _b;              //!< linear equality constraint vector

    VectorXd _lb;             //!< lower bound for each variable (-DRWN_DBL_MAX for unbounded below)
    VectorXd _ub;             //!< upped bound for each variable (DRWN_DBL_MAX for unbounded above)

    VectorXd _x;              //!< current estimate of solution

 public:
    //! construct a problem \f$min. c^T x \text{ s.t. } Ax = b, x \geq 0\f$
    drwnSparseLPSolver(const VectorXd& c, const SparseMatrix<double>& A, const VectorXd& b);
    //! construct a problem \f$min. c^T x \text{ s.t. } Ax = b, l \leq x \leq u\f$
    drwnSparseLPSolver(const VectorXd& c, const SparseMatrix<double>& A, const VectorXd& b,
        const VectorXd& lb, const VectorXd& ub);
    //! destructor
    ~drwnSparseLPSolver() { /* do nothing */ }

    //! initialization of a feasible point
    void initialize(const VectorXd& x);

    //! solve and return objective (solution can be obtained from \ref solution function)
    double solve();

    //! return the current estimate of the solution
    inline const VectorXd& solution() const { return _x; }

    // accessors
    //! return the number of dimensions of the state space
    inline int size() const { return _c.rows(); }    
    //! access the \p i-th dimension of the current solution
    inline double operator[](unsigned i) const { return _x[i]; }

 protected:
    //! returns true if the i-th variable is unbounded
    bool isUnbounded(unsigned i) const { return ((_lb[i] == -DRWN_DBL_MAX) && (_ub[i] == DRWN_DBL_MAX)); }
    //! returns true if a point is strictly within the lower and upper bounds
    bool isWithinBounds(const VectorXd& x) const { return (x.array() > _lb.array()).all() && (x.array() < _ub.array()).all(); }

    //! computes the barrier function for a given assignment
    double barrierFunction(const VectorXd& x) const;
};

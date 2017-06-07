/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnQPSolver.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

#include <cstdlib>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "drwnBase.h"

using namespace std;
using namespace Eigen;

/*!
** \brief Quadratic program solver.
**
** Solves (small scale) quadratic programs of the form:
** \f[
**    \begin{array}{ll}
**    \textrm{minimize (over $x$)} & \frac{1}{2} x^T P x + q^T x + r \\
**    \textrm{subject to}          & Ax = b \\
**                                 & Gx \leq h \\
**                                 & l \leq x \leq u \\
**    \end{array}
** \f]
** where \f$P\f$ is positive semi-definite.
**
** In the general case, the current implementation is based on the method of 
** Hildreth and D'Esops:
**   \li C. Hildreth, A quadratic programming procedure, Naval Research
**       Logistics Quarterly, 1957.
**   \par
**   (see http://onlinelibrary.wiley.com/doi/10.1002/nav.3800040113/pdf)
**   \li D. A. D'Esopo, A convex programming procedure Naval Research
**       Logistics Quarterly 6(1) 1959.
**
** If no equality or general inequality constraints (not including bounds)
** are given, then the problem is solved in primal form. Otherwise, it is
** converted to the dual,
** \f[
**    \begin{array}{ll}
**    \textrm{minimize (over $\lambda, \nu$)} & 
**        \frac{1}{2} (\lambda, \nu)^T \tilde{P} (\lambda, \nu) + \tilde{q}^T (\lambda, \nu) \\
**    \textrm{subject to}          & \nu \geq 0 \\
**    \end{array}
** \f]
** where
** \f[
**     \tilde{P} = \left[ \begin{array}{cccc}
**       A P^{-1} A^T & A P^{-1} G^T & -A P^{-1} & A P^{-1} \\
**       G P^{-1} A^T & G P^{-1} G^T & -G P^{-1} & G P^{-1} \\ 
**       -P^{-1} A^T & -P^{-1} G^T & P^{-1} & -P^{-1} \\ 
**       P^{-1} A^T & P^{-1} G^T & -P^{-1} & P^{-1}
**     \end{array} \right]
**     \quad \textrm{and} \quad
**     \tilde{q} = \left[ \begin{array}{c} b + A P^{-1} q \\ 
**        h + G P^{-1} q \\ -l - P^{-1} q \\ u + P^{-1} q \end{array} \right]
** \f]
** The primal solution is retreived as
** \f$x^\star = -P^{-1}(A^T \lambda^\star + G^T \nu^\star - \nu_{lb}^\star + \nu_{ub}^\star + q)\f$.
**
** If no inequality constraints (including bounds) are given, then the code
** uses an infeasible-start newton method to satisfy the optimality conditions, i.e.,
** \f[
**     \begin{bmatrix} P & A^T \\ A & 0 \end{bmatrix}
**     \begin{bmatrix} x \\ \nu \end{bmatrix} = 
**     \begin{bmatrix} -q \\ b \end{bmatrix}
** \f]
** See:
**   \li S. Boyd and L. Vandenberghe, Convex Optimization, 2004.
**
** Future versions may include specialized algorithms such as:
**   \li V. Franc and V. Hlavac. A Novel Algorithm for Learning Support Vector
**       Machines with Structured Output Spaces, CTU-CMP-2006-04, May, 2006.
**   \li R. J. Vanderbei, LQOQ: An interior point code for quadratic 
**       programming, Technical Report SOR-94-15, Princeton University, 1998.
**
** \warning The QP solver is an experimental feature of the current
** version of \b Darwin.  
**
*/

// drwnQPSolver -------------------------------------------------------------

class drwnQPSolver {
 public:
    // line search parameters
    static double alpha; //!< line search stopping criterion in (0, 1/2)
    static double beta;  //!< line search backtracking parameter in (0, 1)
    
 protected:
    // objective
    MatrixXd _mP;  //!< positive definite quadratic term in the objective function
    VectorXd _q;   //!< linear term in the objective function
    double _r;     //!< constant term in the objective function

    // constraints and bounds
    MatrixXd _mA;  //!< linear equality constraint matrix
    VectorXd _b;   //!< linear equality constraint vector
    MatrixXd _mG;  //!< linear inequality constraint matrix
    MatrixXd _h;   //!< linear inequality constraint vector
    VectorXd _l;   //!< variable lower bounds (box constraint)
    VectorXd _u;   //!< variable upper bounds (box constraint)

    VectorXd _x; //!< current estimate of solution

 public:
    //! default constructor
    drwnQPSolver();
    //! construct an unconstrained QP
    drwnQPSolver(const MatrixXd& P, const VectorXd& q, double r = 0.0);
    virtual ~drwnQPSolver();

    // define problem
    //! set the objective function for the QP (dimensions must agree)
    void setObjective(const MatrixXd& P, const VectorXd& q, double r = 0.0);
    //! set the linear equality constraints for the QP (dimensions must agree)
    void setEqConstraints(const MatrixXd& A, const VectorXd& b);
    //! set the linear inequality constraints for the QP (dimensions must agree)
    void setIneqConstraints(const MatrixXd& G, const VectorXd& h);
    //! set the upper and lower bounds for each variable, i.e., box constraints
    void setBounds(const VectorXd& lb, const VectorXd& ub);    
    //! clear the linear equality constraints
    void clearEqConstraints();
    //! clear the linear inequality constraints
    void clearIneqConstraints();
    //! clear the upper and lower bounds on each variable
    void clearBounds();

    //! initialization (e.g., for warm-start methods)
    virtual void initialize(const VectorXd& x);

    // solve
    //! solve the QP and return the objective value
    virtual double solve();
    //! return the current value of the objective (this is the solution if
    //! the solve() function was previously executed and the problem has not
    //! changed)
    double objective() const;
    //! return the objective value for a given feasible point
    double objective(const VectorXd& x) const;
    //! return the current estimate of the solution
    inline VectorXd solution() const { return _x; }

    // accessors
    //! return the number of dimensions of the state space
    inline int size() const { return _x.rows(); }    
    //! access the \p i-th dimension of the current solution
    inline double operator[](unsigned i) const { return _x[i]; }

 protected:
    // compute gradient: _mP x + _q
    inline VectorXd gradient() const { return _mP * _x + _q; }
    inline VectorXd gradient(const VectorXd& x) const { return _mP * x + _q; }
    
    // check strictly feasibility
    bool isFeasiblePoint(const VectorXd& x) const;

    // special case solvers
    double solveOnlyBounds();
    double solveSingleEquality();
    double solveSimplex();
    double solveNoBounds(); // only equality constraints
    double solveGeneral();

    // line search
    double lineSearchNoBounds(const VectorXd& x, const VectorXd& dx,
        const VectorXd& nu, const VectorXd& dnu) const;
    double lineSearchGeneral(const VectorXd& x, const VectorXd& dx,
        const VectorXd& nu, const VectorXd& dnu) const;

    // solve the system of equations:
    //    | H_x  A^T | | x | = - | c |
    //    |  A   H_y | | y |     | b |
    void solveKKTSystem(const MatrixXd& Hx, const MatrixXd& Hy,
        const MatrixXd& A, const VectorXd& c, const VectorXd& b,
        VectorXd& x, VectorXd& y) const;

    //    | H_x  A^T | | x | = - | c |
    //    |  A   0   | | y |     | b |
    void solveKKTSystem(const MatrixXd& Hx, const MatrixXd& A, 
        const VectorXd& c, const VectorXd& b,
        VectorXd& x, VectorXd& y) const;
};

// drwnLogBarrierQPSolver ---------------------------------------------------

/*!
** Solves (small scale) quadratic programs by adding a log-barrier penalty
** \f[
**    \phi(x) = \sum_k \log(h_k - g^T_k x) + 
**        \sum_i \log(x_i - l_i) + \sum_i \log(u_i - x_i)
** \f]
** and iteratively solving
** \f[
**    \begin{array}{ll}
**    \textrm{minimize (over $x$)} & \frac{1}{2} x^T P x + q^T x + r - \frac{1}{t} \phi(x) \\
**    \textrm{subject to}          & Ax = b
**    \end{array}
** \f]
** for increasing values of \a t.
*/

class drwnLogBarrierQPSolver : public drwnQPSolver {
 public:
    // update parameters
    static double t0; // > 0
    static double mu; // > 1
    
 public:
    //! default constructor
    drwnLogBarrierQPSolver();
    //! construct an unconstrained QP
    drwnLogBarrierQPSolver(const MatrixXd& P, const VectorXd& q, double r = 0.0);
    virtual ~drwnLogBarrierQPSolver();

    //! finds a feasible starting point (or \p returns false if infeasible)
    virtual bool findFeasibleStart();

    //! initialize to feasible point
    virtual void initialize(const VectorXd& x);

    // solve
    virtual double solve();
};    

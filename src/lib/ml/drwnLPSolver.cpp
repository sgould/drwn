/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnLPSolver.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <iostream>
#include <iomanip>

#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

#include "drwnBase.h"
#include "drwnML.h"
#include "drwnLPSolver.h"

using namespace std;
using namespace Eigen;

// drwnLPSolver -------------------------------------------------------------

double drwnLPSolver::t0 = 1.0e-3;
double drwnLPSolver::mu = 10.0;
double drwnLPSolver::eps = 1.0e-6;
unsigned drwnLPSolver::maxiters = 1000;

double drwnLPSolver::alpha = 0.01;
double drwnLPSolver::beta = 0.5;

drwnLPSolver::drwnLPSolver(const VectorXd& c, const MatrixXd& A, const VectorXd& b) :
    _c(c), _A(A), _b(b)
{
    DRWN_ASSERT(_A.rows() == _b.rows());
    DRWN_ASSERT(_A.cols() == _c.rows());

    _lb = VectorXd::Zero(_c.rows());
    _ub = VectorXd::Constant(_c.rows(), DRWN_DBL_MAX);
}

drwnLPSolver::drwnLPSolver(const VectorXd& c, const MatrixXd& A, const VectorXd& b,
    const VectorXd& lb, const VectorXd& ub) : _c(c), _A(A), _b(b), _lb(lb), _ub(ub)
{
    DRWN_ASSERT(_A.rows() == _b.rows());
    DRWN_ASSERT(_A.cols() == _c.rows());
    DRWN_ASSERT(_lb.rows() == _c.rows());
    DRWN_ASSERT(_ub.rows() == _c.rows());
    DRWN_ASSERT((_lb.array() < _ub.array()).all());
}

void drwnLPSolver::initialize(const VectorXd& x)
{
    // ensure that x is strictly feasible
    DRWN_ASSERT_MSG(x.rows() == _c.rows(), "x is the wrong size");
    DRWN_ASSERT_MSG(isWithinBounds(x), "x is not strictly within lower and upper bounds");
    DRWN_ASSERT_MSG(((_A * x - _b).array().abs() < 1.0e-6).all(), "x does not satisfy Ax = b");

    // set x to feasible starting point
    _x = x;
}

double drwnLPSolver::solve()
{
    DRWN_FCN_TIC;
    const int n = _A.cols();
    const int m = _A.rows();

    // find feasible starting point
    if (_x.rows() == 0) {
        _x = VectorXd::Zero(n);
        for (int i = 0; i < n; i++) {
            if (isUnbounded(i)) continue;

            if (_ub[i] == DRWN_DBL_MAX) {
                _x[i] = _lb[i] + 1.0;
            } else if (_lb[i] == -DRWN_DBL_MAX) {
                _x[i] = _ub[i] - 1.0;
            } else {
                _x[i] = 0.5 * (_lb[i] + _ub[i]);
            }
        }
    }

    // initialize kkt system variables
    MatrixXd F = MatrixXd::Zero(n + m, n + m);
    VectorXd g = VectorXd::Zero(n + m);

    F.block(n, 0, m, n) = _A;
    F.block(0, n, n, m) = _A.transpose();

    // initialize dual variables (if needed)
    VectorXd nu = VectorXd::Zero(m);

    // iterate on interior point method
    double t = t0;
    while (1) {
        // determine feasibility
        const bool bFeasible = ((_b - _A * _x).squaredNorm() < eps);
        if (!bFeasible) {
            DRWN_LOG_VERBOSE("...finding feasible point");
        }

        // centering step and update
        for (unsigned iter = 0; iter < maxiters; iter++) {

            //! \todo solve with blockwise elimination
            // construct KKT system, Fx = g
            // | H A^T | | dx | = |  - g   |
            // | A  0  | | w  |   | b - Ax |

            for (int i = 0; i < n; i++) {
                F(i, i) = 0.0;
                if (_ub[i] != DRWN_DBL_MAX) {
                    F(i, i) += 1.0 / ((_ub[i] - _x[i]) * (_ub[i] - _x[i]));
                }
                if (_lb[i] != -DRWN_DBL_MAX) {
                    F(i, i) += 1.0 / ((_x[i] - _lb[i]) * (_x[i] - _lb[i]));
                }
            }

            for (int i = 0; i < n; i++) {
                g[i] = - t * _c[i];
                if (_ub[i] != DRWN_DBL_MAX) {
                    g[i] += 1.0 / (_x[i] - _ub[i]);
                }
                if (_lb[i] != -DRWN_DBL_MAX) {
                    g[i] += 1.0 / (_x[i] - _lb[i]);
                }
            }
            g.tail(m) = _b - _A * _x;

            // check terminating condition
            const double r_primal = g.tail(m).squaredNorm();
            const double r_dual = (_A.transpose() * nu - g.head(n)).squaredNorm();
            //if (!bFeasible && (r_primal + r_dual < drwnLPSolver::eps)) break;
            if (!bFeasible && (r_primal < drwnLPSolver::eps)) break;

            // solve KKT system
            const VectorXd dxnu = F.fullPivLu().solve(g);

            const double lambda_sqr = g.head(n).dot(dxnu.head(n));
            if (bFeasible && (0.5 * lambda_sqr < 1.0e-6)) break;

            if (bFeasible) {
                // feasible line search
                const double f_prev = t * _c.dot(_x) + barrierFunction(_x);
                const double delta_f = alpha * g.dot(dxnu.head(n));

                double step = 1.0;
                while (1) {
                    const VectorXd nx = _x + step * dxnu.head(n);
                    if (isWithinBounds(nx)) {
                        const double f = t * _c.dot(nx) + barrierFunction(nx);
                        if (f - f_prev < step * delta_f) {
                            _x = nx;
                            break;
                        }
                    }
                    step *= beta;
                }
            } else {
                // infeasible start line search
                double step = 1.0;
                while (1) {
                    const VectorXd nx = _x + step * dxnu.head(n);
                    if (isWithinBounds(nx)) {
                        const VectorXd nnu = (1.0 - step) * nu + step * dxnu.tail(m);
                        const double r = (_A.transpose() * nnu - g.head(n)).squaredNorm() +
                            (_b - _A * nx).squaredNorm();
                        if (r <= (1.0 - alpha * step) * (r_primal + r_dual)) {
                            _x = nx;
                            nu = nnu;
                            break;
                        }
                    }
                    step *= beta;
                }
            }
        }

        // check if feasible point was found
        if (!bFeasible && ((_b - _A * _x).squaredNorm() > eps)) {
            DRWN_LOG_WARNING("...could not find a feasible point (residual norm is " <<
                (_b - _A * _x).norm() << ")");
            DRWN_FCN_TOC;
            return DRWN_DBL_MAX;
        }

        DRWN_LOG_VERBOSE("...objective is " << _c.dot(_x));

        // check stopping criteria
        if (m < eps * t) break;

        // update barrier function multiplier
        t *= mu;
    }

    // compute true objective and return
    DRWN_FCN_TOC;
    return _c.dot(_x);
}

double drwnLPSolver::barrierFunction(const VectorXd& x) const
{
    double phi = 0.0;
    for (int i = 0; i < x.rows(); i++) {
        if (_ub[i] != DRWN_DBL_MAX) {
            phi -= log(_ub[i] - x[i]);
        }
        if (_lb[i] != -DRWN_DBL_MAX) {
            phi -= log(x[i] - _lb[i]);
        }
    }
    return phi;
}

// drwnSparseLPSolver -------------------------------------------------------

drwnSparseLPSolver::drwnSparseLPSolver(const VectorXd& c, const SparseMatrix<double>& A,
    const VectorXd& b) : _c(c), _A(A), _b(b)
{
    DRWN_ASSERT(_A.rows() == _b.rows());
    DRWN_ASSERT(_A.cols() == _c.rows());

    _lb = VectorXd::Zero(_c.rows());
    _ub = VectorXd::Constant(_c.rows(), DRWN_DBL_MAX);
}

drwnSparseLPSolver::drwnSparseLPSolver(const VectorXd& c, const SparseMatrix<double>& A,
    const VectorXd& b, const VectorXd& lb, const VectorXd& ub) : 
    _c(c), _A(A), _b(b), _lb(lb), _ub(ub)
{
    DRWN_ASSERT(_A.rows() == _b.rows());
    DRWN_ASSERT(_A.cols() == _c.rows());
    DRWN_ASSERT(_lb.rows() == _c.rows());
    DRWN_ASSERT(_ub.rows() == _c.rows());
    DRWN_ASSERT((_lb.array() < _ub.array()).all());
}

void drwnSparseLPSolver::initialize(const VectorXd& x)
{
    // ensure that x is strictly feasible
    DRWN_ASSERT_MSG(x.rows() == _c.rows(), "x is the wrong size");
    DRWN_ASSERT_MSG(isWithinBounds(x), "x is not strictly within lower and upper bounds");
    DRWN_ASSERT_MSG((VectorXd(_A * x - _b).array().abs() < 1.0e-6).all(), "x does not satisfy Ax = b");

    // set x to feasible starting point
    _x = x;
}

double drwnSparseLPSolver::solve()
{
    DRWN_FCN_TIC;
    const int n = _A.cols();
    const int m = _A.rows();

    // find feasible starting point
    if (_x.rows() == 0) {
        _x = VectorXd::Zero(n);
        for (int i = 0; i < n; i++) {
            if (isUnbounded(i)) continue;

            if (_ub[i] == DRWN_DBL_MAX) {
                _x[i] = _lb[i] + 1.0;
            } else if (_lb[i] == -DRWN_DBL_MAX) {
                _x[i] = _ub[i] - 1.0;
            } else {
                _x[i] = 0.5 * (_lb[i] + _ub[i]);
            }
        }
    }

    // initialize kkt system variables
    SparseMatrix<double> F(n + m, n + m);
    VectorXd g = VectorXd::Zero(n + m);

    vector<Eigen::Triplet<double> > entries;
    entries.reserve(n + 2 * _A.nonZeros());
    for (int i = 0; i < n; i++) {
        entries.push_back(Eigen::Triplet<double>(i, i, 1.0));
    }

    for (int k = 0; k < _A.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(_A, k); it; ++it) {
            entries.push_back(Eigen::Triplet<double>(n + it.row(), it.col(), it.value()));
            entries.push_back(Eigen::Triplet<double>(it.col(), n + it.row(), it.value()));
        }
    }

    F.setFromTriplets(entries.begin(), entries.end());

    // initialize dual variables (if needed)
    VectorXd nu = VectorXd::Zero(m);

    // iterate on interior point method
    double t = drwnLPSolver::t0;
    while (1) {
        // determine feasibility
        const bool bFeasible = ((_b - _A * _x).squaredNorm() < drwnLPSolver::eps);
        if (!bFeasible) {
            DRWN_LOG_VERBOSE("...finding feasible point");
        }

        // centering step and update
        for (unsigned iter = 0; iter < drwnLPSolver::maxiters; iter++) {

            //! \todo solve with blockwise elimination
            // construct KKT system, Fx = g
            // | H A^T | | dx | = |  - g   |
            // | A  0  | | w  |   | b - Ax |

            for (int i = 0; i < n; i++) {
                F.coeffRef(i, i) = 0.0;
                if (_ub[i] != DRWN_DBL_MAX) {
                    F.coeffRef(i, i) += 1.0 / ((_ub[i] - _x[i]) * (_ub[i] - _x[i]));
                }
                if (_lb[i] != -DRWN_DBL_MAX) {
                    F.coeffRef(i, i) += 1.0 / ((_x[i] - _lb[i]) * (_x[i] - _lb[i]));
                }
            }

            for (int i = 0; i < n; i++) {
                g[i] = - t * _c[i];
                if (_ub[i] != DRWN_DBL_MAX) {
                    g[i] += 1.0 / (_x[i] - _ub[i]);
                }
                if (_lb[i] != -DRWN_DBL_MAX) {
                    g[i] += 1.0 / (_x[i] - _lb[i]);
                }
            }
            g.tail(m) = _b - _A * _x;

            // check terminating condition
            const double r_primal = g.tail(m).squaredNorm();
            const double r_dual = (_A.transpose() * nu - g.head(n)).squaredNorm();
            //if (!bFeasible && (r_primal + r_dual < drwnLPSolver::eps)) break;
            if (!bFeasible && (r_primal < drwnLPSolver::eps)) break;

            // solve KKT system
            SimplicialLDLT<SparseMatrix<double> > chol(F);
            const VectorXd dxnu = chol.solve(g);

            const double lambda_sqr = g.head(n).dot(dxnu.head(n));
            if (bFeasible && (0.5 * lambda_sqr < 1.0e-6)) break;

            if (bFeasible) {
                // feasible line search
                const double f_prev = t * _c.dot(_x) + barrierFunction(_x);
                const double delta_f = drwnLPSolver::alpha * g.dot(dxnu.head(n));

                double step = 1.0;
                while (1) {
                    const VectorXd nx = _x + step * dxnu.head(n);
                    if (isWithinBounds(nx)) {
                        const double f = t * _c.dot(nx) + barrierFunction(nx);
                        if (f - f_prev < step * delta_f) {
                            _x = nx;
                            break;
                        }
                    }
                    step *= drwnLPSolver::beta;
                }
            } else {
                // infeasible start line search
                DRWN_LOG_DEBUG("...iteration " << iter << ", primal residual " << r_primal
                    << ", dual residual " << r_dual);

                double step = 1.0;
                while (1) {
                    const VectorXd nx = _x + step * dxnu.head(n);
                    if (isWithinBounds(nx)) {
                        const VectorXd nnu = (1.0 - step) * nu + step * dxnu.tail(m);
                        const double r = (_A.transpose() * nnu - g.head(n)).squaredNorm() +
                            (_b - _A * nx).squaredNorm();
                        if (r <= (1.0 - drwnLPSolver::alpha * step) * (r_primal + r_dual)) {
                            _x = nx;
                            nu = nnu;
                            break;
                        }
                    }
                    step *= drwnLPSolver::beta;
                }
            }
        }

        // check if feasible point was found
        if (!bFeasible && ((_b - _A * _x).squaredNorm() > drwnLPSolver::eps)) {
            DRWN_LOG_WARNING("...could not find a feasible point (residual norm is " <<
                (_b - _A * _x).norm() << ")");
            DRWN_FCN_TOC;
            return DRWN_DBL_MAX;
        }

        DRWN_LOG_VERBOSE("...objective is " << _c.dot(_x));

        // check stopping criteria
        if (m < drwnLPSolver::eps * t) break;

        // update barrier function multiplier
        t *= drwnLPSolver::mu;
    }

    // compute true objective and return
    DRWN_FCN_TOC;
    return _c.dot(_x);
}

double drwnSparseLPSolver::barrierFunction(const VectorXd& x) const
{
    double phi = 0.0;
    for (int i = 0; i < x.rows(); i++) {
        if (_ub[i] != DRWN_DBL_MAX) {
            phi -= log(_ub[i] - x[i]);
        }
        if (_lb[i] != -DRWN_DBL_MAX) {
            phi -= log(x[i] - _lb[i]);
        }
    }
    return phi;
}

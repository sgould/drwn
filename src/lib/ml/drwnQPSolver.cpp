/******************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnQPSolver.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdio>
#include <iostream>

#include "drwnBase.h"
#include "drwnQPSolver.h"

using namespace std;
using namespace Eigen;

// drwnQPSolver -------------------------------------------------------------

double drwnQPSolver::alpha = 0.3;
double drwnQPSolver::beta = 0.5;

drwnQPSolver::drwnQPSolver() :
    _r(0.0)
{
    // do nothing
}

drwnQPSolver::drwnQPSolver(const MatrixXd& P, const VectorXd& q, double r)
{
    setObjective(P, q, r);
}

drwnQPSolver::~drwnQPSolver()
{
    // do nothing
}

// define problem
void drwnQPSolver::setObjective(const MatrixXd& P, const VectorXd& q, double r)
{
    DRWN_ASSERT((P.rows() == P.cols()) && (P.cols() == q.rows()));
#if 1
    // check that P is positive definite
    Eigen::LDLT<MatrixXd> cholesky(P);
    DRWN_ASSERT_MSG(cholesky.isPositive(), "P must be positive definite");
#endif

    _mP = P;
    _q = q;
    _r = r;

    _x = VectorXd::Zero(_q.rows());

    _l = VectorXd::Constant(_x.rows(), -DRWN_DBL_MAX);
    _u = VectorXd::Constant(_x.rows(), DRWN_DBL_MAX);
}

void drwnQPSolver::setEqConstraints(const MatrixXd& A, const VectorXd& b)
{
    DRWN_ASSERT((A.rows() == b.rows()) && (A.cols() == _mP.rows()));

    _mA = A;
    _b = b;
}

void drwnQPSolver::setIneqConstraints(const MatrixXd& G, const VectorXd& h)
{
    DRWN_ASSERT((G.rows() == h.rows()) && (G.cols() == _mP.rows()));

    _mG = G;
    _h = h;
}

void drwnQPSolver::setBounds(const VectorXd& lb, const VectorXd& ub)
{
    DRWN_ASSERT((lb.rows() == _mP.rows()) && (ub.rows() == _mP.rows()));

    _l = lb;
    _u = ub;
}

void drwnQPSolver::clearEqConstraints()
{
    _mA.resize(0, 0);
    _b.resize(0);
}

void drwnQPSolver::clearIneqConstraints()
{
    _mG.resize(0, 0);
    _h.resize(0, 0);
}

void drwnQPSolver::clearBounds()
{
    _l = VectorXd::Constant(_x.rows(), -DRWN_DBL_MAX);
    _u = VectorXd::Constant(_x.rows(), DRWN_DBL_MAX);
}

void drwnQPSolver::initialize(const VectorXd& x)
{
    DRWN_ASSERT(x.rows() == _x.rows());
    _x = x;
}

double drwnQPSolver::solve()
{
    // invoke specialized methods
    if ((_b.rows() == 0) && (_h.rows() == 0)) {
        return solveOnlyBounds();
    }

    DRWN_LOG_WARNING_ONCE("drwnQPSolver::solve() has not been debugged");

    if ((_b.rows() == 1) && (_h.rows() == 0)) {
        return solveSingleEquality();
    }

    if ((_b.rows() != 0) && (_h.rows() == 0) &&
        ((_mA.array() == 1.0) + (_mA.array() == 0.0)).all()) {
            return solveSimplex();
    }

    // general solvers
    bool bBounds = (_h.rows() != 0);
    for (int i = 0; i < _x.size(); i++) {
        if ((_l[i] != -DRWN_DBL_MAX) || (_u[i] != DRWN_DBL_MAX)) {
            bBounds = true;
            break;
        }
    }

    return bBounds ? solveGeneral() : solveNoBounds();
}

double drwnQPSolver::objective() const
{
    return objective(_x);
}

double drwnQPSolver::objective(const VectorXd& x) const
{
    DRWN_ASSERT(x.rows() == _mP.rows());

    return (0.5 * (x.transpose() * _mP * x)[0] + _q.dot(x) + _r);
}

// protected functions ------------------------------------------------------

bool drwnQPSolver::isFeasiblePoint(const VectorXd& x) const
{
    DRWN_ASSERT(x.rows() == _x.rows());

    if ((x.array() <= _l.array()).any()) return false;
    if ((x.array() >= _u.array()).any()) return false;
    if ((_mA.rows() != 0) && ((_mA * x - _b).norm() > DRWN_EPSILON)) return false;
    if ((_mG.rows() != 0) && ((_h - _mG * x).array() <= 0.0).any()) return false;

    return true;
}

// special case solvers
double drwnQPSolver::solveOnlyBounds()
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(_b.rows() == 0);

    // find initial good solution
#if 0
    _mP.ldlt().solve(-_q, &_x);
    bool bConstraintsViolated = false;
#else
    _x = VectorXd::Zero(_mP.rows());
    bool bConstraintsViolated = true;
#endif

    for (int i = 0; i < this->size(); i++) {
        if (_x[i] < _l[i]) {
            _x[i] = _l[i];
            bConstraintsViolated = true;
        } else if (_x[i] > _u[i]) {
            _x[i] = _u[i];
            bConstraintsViolated = true;
        }
    }

    // iterate until convergence
    double bestObj = this->objective();
    int nIterations = 0;
    while (bConstraintsViolated) {
        nIterations += 1;

        // gauss-seidel
        for (int i = 0; i < this->size(); i++) {
            double wi = (_mP(i,i) * _x[i] - _mP.row(i).dot(_x) - _q[i]) / _mP(i,i);
            _x[i] = std::min(_u[i], std::max(_l[i], wi));
        }

        double obj = this->objective();
        DRWN_LOG_DEBUG("iteration " << nIterations << ", objective " << obj << " (delta: " << bestObj - obj << ")");
        if ((bestObj - obj) < DRWN_EPSILON)
            break;
        bestObj = obj;
    }

    DRWN_FCN_TOC;
    return objective(_x);
}

double drwnQPSolver::solveSingleEquality()
{
    DRWN_FCN_TIC;
    DRWN_ASSERT(_b.rows() == 1);
    // need _x to be feasible?

    while (1) {
        // compute gradient and objective
        VectorXd g = gradient(_x);
        DRWN_LOG_VERBOSE("... f = " << (0.5 * g.dot(_x) + _q.dot(_x)));

        // find most violated pair of variables
        int u = -1;
        int v = -1;
        double Fu = DRWN_DBL_MAX;
        double Fv = -DRWN_DBL_MAX;
        for (int i = 0; i < _x.rows(); i++) {
            double Fi = g[i] / _mA(i);
            if ((_l[i] < _x[i]) && (_x[i] < _u[i])) { // set I_0
                if (Fu > Fi) { Fu = Fi; u = i; }
                if (Fv < Fi) { Fv = Fi; v = i; }
            } else if (((_mA(i) > 0.0) && (_x[i] == _l[i])) ||
                ((_mA(i) < 0.0) && (_x[i] == _u[i]))) { // set I_1 or I_2
                if (Fu > Fi) { Fu = Fi; u = i; }
            } else if (((_mA(i) > 0.0) && (_x[i] == _u[i])) ||
                ((_mA(i) < 0.0) && (_x[i] == _l[i]))) { // set I_3 or I_4
                if (Fv < Fi) { Fv = Fi; v = i; }
            }
        }

        // check KKT conditions
        if (Fv - Fu <= DRWN_EPSILON) {
            DRWN_LOG_VERBOSE("relaxed KKT conditions satisfied");
            break;
        }

        // SMO update
        double tau_lb, tau_ub;
        if (_mA(u) > 0.0) {
            tau_lb = _mA(u) * (_l[u] - _x[u]);
            tau_ub = _mA(u) * (_u[u] - _x[u]);
        } else {
            tau_ub = _mA(u) * (_l[u] - _x[u]);
            tau_lb = _mA(u) * (_u[u] - _x[u]);
        }

        if (_mA(v) > 0.0) {
            tau_lb = std::max(tau_lb, _mA(v) * (_x[v] - _u[v]));
            tau_ub = std::min(tau_ub, _mA(v) * (_x[v] - _l[v]));
        } else {
            tau_lb = std::max(tau_lb, _mA(v) * (_x[v] - _l[v]));
            tau_ub = std::min(tau_ub, _mA(v) * (_x[v] - _u[v]));
        }

        double tau = (Fv - Fu) /
            (_mP(u,u) / (_mA(u) * _mA(u)) + _mP(v,v) / (_mA(v) * _mA(v)) - 2.0 * _mP(u,v) / (_mA(u) * _mA(v)));
        tau = std::min(std::max(tau, tau_lb), tau_ub);

        _x[u] += tau / _mA(u);
        _x[v] += tau / _mA(v);
    }

    DRWN_FCN_TOC;
    return objective(_x);
}

double drwnQPSolver::solveSimplex()
{
    DRWN_FCN_TIC;

    //! \todo implement specialized simplex solver
    DRWN_TODO;

    DRWN_FCN_TOC;
    return solveGeneral();
}

double drwnQPSolver::solveNoBounds()
{
    DRWN_FCN_TIC;

    // initialize primal and dual variables
    _x = VectorXd::Zero(_q.rows());
    VectorXd nu = VectorXd::Zero(_mA.rows());

    VectorXd dx, dnu;

    double r = DRWN_DBL_MAX;
    while (r > 0.5 * DRWN_EPSILON) {
        solveKKTSystem(_mP, _mA, _mP * _x + _q + _mA.transpose() * nu, _mA * _x - _b, dx, dnu);
        double t = lineSearchNoBounds(_x, dx, nu, dnu);
        _x = _x + t * dx;
        nu = nu + t * dnu;

        r = (_mP * _x + _q + _mA.transpose() * nu).norm() + (_mA * _x - _b).norm();
        DRWN_LOG_DEBUG("..." << r << " (" << t << ")");
    }

    DRWN_FCN_TOC;
    return objective(_x);
}

double drwnQPSolver::solveGeneral()
{
    DRWN_FCN_TIC;

    // create dual problem
    // \todo only include valid upper and lower bounds
    bool bHasLowerBounds = (_l.array() != -DRWN_DBL_MAX).any();
    bool bHasUpperBounds = (_u.array() != DRWN_DBL_MAX).any();

    // dual problem size
    int n = _mA.rows() + _mG.rows() + (bHasLowerBounds ? _l.rows() : 0) +
        (bHasUpperBounds ? _u.rows() : 0);
    DRWN_LOG_VERBOSE("solving dual problem of size " << n);

    VectorXd x_unconst(_mP.ldlt().solve(-_q));
    DRWN_LOG_DEBUG("unconstrained solution " << x_unconst.transpose());

    MatrixXd invP = _mP.inverse();
    MatrixXd stacked(n, _mP.rows());
    VectorXd tq(n);

    if (_mA.rows() != 0) {
        stacked.block(0, 0, _mA.rows(), _mA.cols()) = _mA;
        tq.segment(0, _mA.rows()) = _b - _mA * x_unconst;
    }
    if (_mG.rows() != 0) {
        stacked.block(_mA.rows(), 0, _mG.rows(), _mG.cols()) = _mG;
        tq.segment(_mA.rows(), _mG.rows()) = _h - _mG * x_unconst;
    }
    if (bHasLowerBounds) {
        stacked.block(_mA.rows() + _mG.rows(), 0, _mP.rows(), _mP.rows()) = -1.0 * MatrixXd::Identity(_mP.rows(), _mP.cols());
        tq.segment(_mA.rows() + _mG.rows(), _mP.rows()) = -1.0 * (_l - x_unconst);
    }
    if (bHasUpperBounds) {
        stacked.block(_mA.rows() + _mG.rows() + (bHasLowerBounds ? _mP.rows() : 0),
            0, _mP.rows(), _mP.rows()) = MatrixXd::Identity(_mP.rows(), _mP.cols());
        tq.tail(_mP.rows()) = _u - x_unconst;
    }

    MatrixXd tP = stacked * invP * stacked.transpose();
    double tr = -0.5 * _q.dot(x_unconst) - _r;

    // lower bounds
    VectorXd lb(n);
    lb << VectorXd::Constant(_mA.rows(), -DRWN_DBL_MAX), VectorXd::Zero(n - _mA.rows());

    // create new solver and solve dual
    drwnQPSolver solver(tP, tq, tr);
    solver.setBounds(lb, VectorXd::Constant(n, DRWN_DBL_MAX));
    solver.solve();

    // retreive primal solution
    VectorXd y = solver.solution();

    VectorXd t = _q;
    if (_mA.rows() != 0) {
        t += _mA.transpose() * y.segment(0, _mA.rows());
    }
    if (_mG.rows() != 0) {
        t += _mG.transpose() * y.segment(_mA.rows(), _mG.rows());
    }
    if (bHasLowerBounds) {
        t -= y.segment(_mA.rows() + _mG.rows(), _mP.rows());
    }
    if (bHasUpperBounds) {
        t += y.tail(_mP.rows());
    }

    _x = -1.0 * invP * t;
    DRWN_ASSERT(_x.rows() == _mP.rows());

    DRWN_FCN_TOC;
    return objective(_x);
}

// line search
double drwnQPSolver::lineSearchNoBounds(const VectorXd& x, const VectorXd& dx,
    const VectorXd& nu, const VectorXd& dnu) const
{
    DRWN_FCN_TIC;
    DRWN_ASSERT((alpha > 0.0) && (alpha < 0.5));
    DRWN_ASSERT((beta > 0.0) && (beta < 1.0));

    double t = 1.0;
    double r = (_mP * x + _q + _mA.transpose() * nu).norm() + (_mA * x - _b).norm();

    while (t > DRWN_DBL_MIN) {
        double r2 = ( _mP * (x + t * dx) + _q + _mA.transpose() * (nu + t * dnu)).norm() +
            (_mA * (x + t * dx) - _b).norm();

        if (r2 <= alpha * t * r)
            break;

        t *= beta;
    }

    DRWN_FCN_TOC;
    return t;
}

double drwnQPSolver::lineSearchGeneral(const VectorXd& x, const VectorXd& dx,
    const VectorXd& nu, const VectorXd& dnu) const
{
    //! \todo implement
    DRWN_TODO;
    return 0.0;
}

/*
** solve the system of equations:
**    | H_x  A^T | | x | = - | c |
**    |  A   H_y | | y |     | b |
*/
void drwnQPSolver::solveKKTSystem(const MatrixXd& Hx, const MatrixXd& Hy,
    const MatrixXd& A, const VectorXd& c, const VectorXd& b,
    VectorXd& x, VectorXd& y) const
{
    // check input
    DRWN_ASSERT(Hx.rows() == Hx.cols());
    DRWN_ASSERT(Hy.rows() == Hy.cols());
    DRWN_ASSERT((A.rows() == Hy.rows()) && (A.cols() == Hx.cols()));
    DRWN_ASSERT((Hx.rows() == c.rows()) && (A.rows() == b.rows()));

    // eliminate x = -H_x^{-1} (A^T y + c)
    Eigen::LDLT<MatrixXd> ldl(Hx);
    MatrixXd invHxAt(Hx.rows(), A.rows());
    VectorXd invHxc(Hx.rows());

    for (int i = 0; i < A.rows(); i++) {
        invHxAt.col(i) = ldl.solve(A.row(i).transpose());
    }
    invHxc = ldl.solve(c);

    // solve for y
    MatrixXd G = Hy - A * invHxAt;
    y = G.ldlt().solve(A * invHxc - b);

    // solve for x
    x = -invHxAt * y - invHxc;
}

/*
** solve the system of equations:
**    | H_x  A^T | | x | = - | c |
**    |  A   0   | | y |     | b |
*/
void drwnQPSolver::solveKKTSystem(const MatrixXd& Hx,
    const MatrixXd& A, const VectorXd& c, const VectorXd& b,
    VectorXd& x, VectorXd& y) const
{
    // check input
    DRWN_ASSERT((Hx.rows() == Hx.cols()) && (A.cols() == Hx.cols()));
    DRWN_ASSERT((Hx.rows() == c.rows()) && (A.rows() == b.rows()));

    // eliminate x = -H_x^{-1} (A^T y + c)
    Eigen::LDLT<MatrixXd> ldl(Hx);
    MatrixXd invHxAt(Hx.rows(), A.rows());
    VectorXd invHxc(Hx.rows());

    for (int i = 0; i < A.rows(); i++) {
        invHxAt.col(i) = ldl.solve(A.row(i).transpose());
    }
    invHxc = ldl.solve(c);

    // solve for y
    MatrixXd G = A * invHxAt;
    y = G.ldlt().solve(b - A * invHxc);

    // solve for x
    x = -invHxAt * y - invHxc;
}

// drwnLogBarrierQPSolver ---------------------------------------------------

double drwnLogBarrierQPSolver::t0 = 5.0;
double drwnLogBarrierQPSolver::mu = 20.0;

drwnLogBarrierQPSolver::drwnLogBarrierQPSolver() : drwnQPSolver()
{
    // do nothing
}

drwnLogBarrierQPSolver::drwnLogBarrierQPSolver(const MatrixXd& P, const VectorXd& q, double r) :
    drwnQPSolver(P, q, r)
{
    // do nothing
}

drwnLogBarrierQPSolver::~drwnLogBarrierQPSolver()
{
    // do nothing
}

// initialize
bool drwnLogBarrierQPSolver::findFeasibleStart()
{
    if (_mG.rows() == 0) {
        if (_mA.rows() == 0) {
            _x.setZero();
        } else {
            _x = _mA.ldlt().solve(_b);
        }
        return true;
    }

    // check is solution is already feasible
    if (isFeasiblePoint(_x))
        return true;

    // solves the problem
    // minimize s
    // subject to Gx - h <= s
    // TODO: need an LP solver

    DRWN_FCN_TIC;
    DRWN_LOG_DEBUG("starting infeasible start method...");

    VectorXd q(_x.rows() + 1); q << VectorXd::Zero(_x.rows()), 1.0;
    drwnLogBarrierQPSolver solver(1.0e-3 * MatrixXd::Identity(_x.rows() + 1, _x.rows() + 1), q);

    if (_mA.rows() != 0) {
        MatrixXd A(_mA.rows(), _x.rows() + 1);
        A.topLeftCorner(_mA.rows(), _mA.cols()) = _mA;
        A.col(_x.rows()).setConstant(0.0);
        solver.setEqConstraints(A, _b);
    }
    
    MatrixXd G(_h.rows(), _x.rows() + 1);
    G.topLeftCorner(_mG.rows(), _mG.cols()) = _mG;
    G.col(_x.rows()).setConstant(-1.0);
    solver.setIneqConstraints(G, _h);

    VectorXd lb(_x.rows() + 1); lb << _l, -DRWN_DBL_MAX;
    VectorXd ub(_x.rows() + 1); ub << _u, DRWN_DBL_MAX;
    solver.setBounds(lb, ub);
    
    double s = 1.0 - _h.minCoeff();
    for (int i = 0; i < _x.rows(); i++) {
        s = std::max(s, std::max(1.0 - _u[i], _l[i] - 1.0));
    }

    VectorXd x = VectorXd::Zero(_x.rows() + 1);
    x[_x.rows()] = s;

    solver.initialize(x);
    solver.solve();

    _x = solver.solution().head(_x.rows());

    DRWN_FCN_TOC;
    return true;
}

void drwnLogBarrierQPSolver::initialize(const VectorXd& x)
{
    // point needs to be strictly feasible
    DRWN_ASSERT(isFeasiblePoint(x));
    _x = x;
}

// solve
double drwnLogBarrierQPSolver::solve()
{
    DRWN_ASSERT_MSG(_mA.rows() == 0, "not yet implemented for equality constraints");
    if (_mG.rows() == 0) {
        return drwnQPSolver::solve();
    }

    DRWN_FCN_TIC;
    double t = t0;

    while (1) {
        // centering step
        DRWN_LOG_DEBUG("centering step with t = " << t << "...");
        while (1) {
            VectorXd d = (_h - _mG * _x).array().inverse();
            double J = t * objective(_x) + d.array().log().sum();
            VectorXd Jx = t * gradient(_x) + _mG.transpose() * d;

            for (int i = 0; i < _x.rows(); i++) {
                if (_l[i] != -DRWN_DBL_MAX) {
                    J -= log(_x[i] - _l[i]);
                    Jx[i] -= 1.0 / (_x[i] - _l[i]);
                }
                if (_u[i] != DRWN_DBL_MAX) {
                    J -= log(_u[i] - _x[i]);
                    Jx[i] += 1.0 / (_u[i] - _x[i]);
                }
            }

            if (Jx.norm() < 1.0e-6) {
                DRWN_LOG_DEBUG("Jx = " << Jx.transpose());
                break;
            }

            MatrixXd Jxx = t * _mP;
            for (int m = 0; m < _mG.rows(); m++) {
                Jxx += d[m] * d[m] * _mG.row(m).transpose() * _mG.row(m);
            }

            for (int i = 0; i < _x.rows(); i++) {
                if (_l[i] != -DRWN_DBL_MAX) {
                    Jxx(i,i) += 1.0 / ((_x[i] - _l[i]) * (_x[i] - _l[i]));
                }
                if (_u[i] != DRWN_DBL_MAX) {
                    Jx(i,i) += 1.0 / ((_u[i] - _x[i]) * (_u[i] - _x[i]));
                }
            }

            // newton direction
            VectorXd dx = alpha * Jxx.ldlt().solve(Jx);
            for (int i = 0; i < dx.rows(); i++) {
                DRWN_ASSERT_MSG(isfinite(dx[i]), dx.transpose());
            }

            // line search
            double J_new = J;
            VectorXd x_new = _x;
            while (1) {
                x_new = _x - dx;
                d = _h - _mG * x_new;
                if ((d.array() > 0.0).all() && 
                    (x_new.array() >= _l.array()).all() && 
                    (x_new.array() <= _u.array()).all()) {
                    J_new = t * objective(x_new) - d.array().log().sum();
                    DRWN_ASSERT_MSG(isfinite(J_new), x_new.transpose() << "; " << d.transpose());
                    for (int i = 0; i < _x.rows(); i++) {
                        if (_l[i] != -DRWN_DBL_MAX) {
                            J_new -= log(x_new[i] - _l[i]);
                        }
                        if (_u[i] != DRWN_DBL_MAX) {
                            J_new -= log(_u[i] - x_new[i]);
                        }
                    }
                    if (J_new <= J - dx.norm()) break;
                }
                dx *= beta;
                if (dx.norm() < 1.0e-12) break;
            }

            // update
            _x = x_new;

            if (J - J_new < 1.0e-6 * std::min(1.0, fabs(J))) {
                DRWN_LOG_DEBUG("J = " << J << ", J_new = " << J_new << ", dJ = " << (J - J_new));
                break;
            }
        }

        DRWN_LOG_VERBOSE("..." << objective());

        // stopping criterion
        if (_mG.rows() / t < DRWN_EPSILON)
            break;

        // increase t
        t *= mu;
    }

    DRWN_FCN_TOC;
    return objective(_x);
}

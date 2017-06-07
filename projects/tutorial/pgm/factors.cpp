/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    factors.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnPGM.h"

using namespace std;

// dot product factor operations ---------------------------------------------

class DotProductOp {
protected:
    double *_result;
    const drwnTableFactor * const _A;
    const drwnTableFactor * const _B;
    drwnTableFactorMapping _mappingA;
    drwnTableFactorMapping _mappingB;

public:
    DotProductOp(double *result, const drwnTableFactor *A, const drwnTableFactor *B);
    ~DotProductOp() { /* do nothing */ };

    void execute();
};

DotProductOp::DotProductOp(double *result, const drwnTableFactor *A,
    const drwnTableFactor *B) : _result(result), _A(A), _B(B)
{
    drwnClique c;
    c.insert(_A->getOrderedVars().begin(), _A->getOrderedVars().end());
    c.insert(_B->getOrderedVars().begin(), _B->getOrderedVars().end());
    vector<int> vars(c.begin(), c.end());

    _mappingA = drwnTableFactorMapping(vars, _A->getOrderedVars(), _A->getUniverse());
    _mappingB = drwnTableFactorMapping(vars, _B->getOrderedVars(), _B->getUniverse());
}

void DotProductOp::execute()
{
    drwnTableFactorMapping::iterator ia(_mappingA.begin());
    drwnTableFactorMapping::iterator ib(_mappingB.begin());
    *_result = 0.0;
    while (ia != _mappingA.end()) {
        *_result += (*_A)[*ia++] * (*_B)[*ib++];
    }
}

// main ----------------------------------------------------------------------

int main()
{
    // define variables
    drwnVarUniversePtr universe(new drwnVarUniverse());
    universe->addVariable(3, "a");
    universe->addVariable(3, "b");

    universe->print();

    // dot-product between unary factors over same variable
    drwnTableFactor psiA1(universe);
    psiA1.addVariable("a");
    psiA1[0] = 1.0; psiA1[1] = 2.0; psiA1[2] = 3.0;

    drwnTableFactor psiA2(universe);
    psiA2.addVariable("a");
    psiA2[0] = -1.0; psiA2[1] = 0.0; psiA2[2] = 1.0;

    double result;
    DotProductOp(&result, &psiA1, &psiA2).execute();

    DRWN_LOG_MESSAGE("dot-product between A:[1 2 3] and A:[-1 0 1] is " << result);

    // dot-product between unary factors over different variables
    drwnTableFactor psiB(universe);
    psiB.addVariable("b");

    psiB[0] = -1.0; psiB[1] = 0.0; psiB[2] = 1.0;
    DotProductOp(&result, &psiA1, &psiB).execute();
    DRWN_LOG_MESSAGE("dot-product between A:[1 2 3] and B:[-1 0 1] is " << result);

    psiB[0] = 1.0; psiB[1] = 0.0; psiB[2] = 0.0;
    DotProductOp(&result, &psiA1, &psiB).execute();
    DRWN_LOG_MESSAGE("dot-product between A:[1 2 3] and B:[1 0 0] is " << result);

    // dot-product between unary and pairwise factors
    drwnTableFactor psiAB(universe);
    psiAB.addVariables("b", "a", NULL);

    const double dataAB[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    psiAB.copy(dataAB);

    DotProductOp(&result, &psiAB, &psiB).execute();
    DRWN_LOG_MESSAGE("dot-product between A,B:[1 2 3; 4 5 6; 7 8 9] and B:[1 0 0] is " << result);

    DotProductOp(&result, &psiAB, &psiA2).execute();
    DRWN_LOG_MESSAGE("dot-product between A,B:[1 2 3; 4 5 6; 7 8 9] and A:[-1 0 1] is " << result);

    return 0;
}

#include "factorAdditionOp.h"

// FactorAdditionOp  ----------------------------------------------------------
FactorAdditionOp::FactorAdditionOp(drwnGeneralFactor *target, const drwnGeneralFactor *A,
		const drwnGeneralFactor *B) : drwnFactorNAryOp(target, A, B)
{
	target->setOps(A, B);
	vector<int> ordTargetVars = target->getOrderedVars();
	vector<int> ordVarsA = _generalFactors.front()->getOrderedVars();
	vector<int> ordVarsB = _generalFactors[1]->getOrderedVars();

	set<int> varsA(ordVarsA.begin(), ordVarsA.end());
	set<int> varsB(ordVarsB.begin(), ordVarsB.end());
	drwnFactorStorageType storTypeA = _generalFactors.front()->getStorageType();
	drwnFactorStorageType storTypeB = _generalFactors[1]->getStorageType();

	if ((storTypeA == EVAL) || (storTypeB == EVAL) || (varsA != varsB)) {
		_intendedStorageType = EVAL;
		return;
	}

	/* now have that varsA == varsB */
	if ((storTypeA == SPARSE) && (storTypeB == SPARSE)) {
		set<int> targetVars(ordTargetVars.begin(), ordTargetVars.end());	
		_intendedStorageType = SPARSE;

		for (set<int>::iterator si = targetVars.begin(); si != targetVars.end(); si++) {
			vector<int>::iterator it = find(ordVarsA.begin(), ordVarsA.end(), *si);
			_targetToA.push_back(it - ordVarsA.begin());
			it = find(ordVarsB.begin(), ordVarsB.end(), *si);
			_targetToB.push_back(it - ordVarsB.begin());
		}

		return;
	}

	if ((storTypeA == DENSE) && (storTypeB == DENSE)) {
		_intendedStorageType = DENSE;
		return;
	}

	/* have that varsA == varsB and adding dense to sparse */
	_intendedStorageType = EVAL;	/* temporary hack */
	/* _intendedStorageType = DENSE; */
	/* TODO: convert sparse factor to dense factor */
}

FactorAdditionOp::FactorAdditionOp(drwnGeneralFactor *target,
		const vector<const drwnGeneralFactor *>& A) :
    drwnFactorNAryOp(target, A)
{	// do nothing
}

void FactorAdditionOp::execute()
{
	if (_generalFactors.size() == 2) {
		if (_intendedStorageType == EVAL) {
			_generalTarget->setStorageType(EVAL);
			return;
		} else if (_intendedStorageType == SPARSE) {
			/* TODO: test this */
#if 0
			vector<int> vars = _generalTarget->getOrderedVars();
			drwnSparseFactor *fac1 = _A->getSparseFactor();
			drwnSparseFactor *fac2 = _B->getSparseFactor();
			drwnSparseFactor *target = _generalTarget->getSparseFactor();
			map<vector<int>, double> assignmtsA = fac1->getAssignments();
			map<vector<int>, double> assignmtsB = fac2->getAssignments();

			for (map< vector<int>, double >::iterator mi = assignmtsA.begin();
					mi != assignmtsA.end(); mi++) {
				vector<int> vals(vars.size());

				for (int i = 0; i < vars.size(); i++) {
					vals[_targetToA[i]] = (*mi).first[_targetToA[i]];
				}

				drwnPartialAssignment dpa;
				target->setValueOf(dpa, (*mi).second);
			}

			for (map< vector<int>, double >::iterator mi = assignmtsB.begin();
					mi != assignmtsA.end(); mi++) {
				vector<int> vals(vars.size());

				for (int i = 0; i < vars.size(); i++) {
					vals[_targetToB[i]] = (*mi).first[_targetToB[i]];
				}

				drwnPartialAssignment dpa;
				target->setValueOf(dpa, target->getValueOf(dpa) + (*mi).second);
			}
#endif

			return;
		}

		/* dealing with dense factors */
		drwnTableFactor *target = _generalTarget->getTableFactor();
		drwnFactorAdditionOp dfao(target, _generalFactors[0]->getTableFactor(),
				_generalFactors[1]->getTableFactor());
		dfao.execute();
		_generalTarget = new drwnGeneralFactor(*target, _generalTarget->THRESHOLD);
		return;
	}

	drwnGeneralFactor *prevSum = _generalFactors.front()->clone();

	for (int fi = 1; fi < _generalFactors.size(); fi++) {
		drwnTableFactor tf(_generalTarget->getUniverse());
		drwnGeneralFactor *partSum = (fi == (_generalFactors.size() - 1)) ?
				_generalTarget :
				new drwnGeneralFactor(tf, _generalTarget->THRESHOLD);

		FactorAdditionOp fao(partSum, prevSum, _generalFactors[fi]);
		fao.execute();
		prevSum = partSum;
	}
}

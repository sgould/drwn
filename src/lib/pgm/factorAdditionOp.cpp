#include "factorAdditionOp.h"

// FactorAdditionOp  ----------------------------------------------------------
FactorAdditionOp::FactorAdditionOp(drwnGeneralFactor *target, const drwnGeneralFactor *A,
		const drwnGeneralFactor *B) : drwnFactorNAryOp(target, A, B)
{
	target->setOps(A, B);
	vector<int> ordTargetVars = target->getOrderedVars();

	if (ordTargetVars.empty()) {
		vector<int> varsA = A->getOrderedVars();
		vector<int> varsB = B->getOrderedVars();
		set<int> AunionB;

		set_union(varsA.begin(), varsA.end(), varsB.begin(), varsB.end(),
			inserter(AunionB, AunionB.begin()));

		for (set<int>::iterator si = AunionB.begin(); si != AunionB.end(); si++) {
			target->addVariable(*si);
		}
	}

	vector<int> ordVarsA = A->getOrderedVars();
	vector<int> ordVarsB = B->getOrderedVars();
	set<int> varsA(ordVarsA.begin(), ordVarsA.end());
	set<int> varsB(ordVarsB.begin(), ordVarsB.end());

	if ((A->getStorageType() == EVAL) || (B->getStorageType() == EVAL) ||
		(varsA != varsB)) {
		_intendedStorageType = EVAL;
		return;
	}

	/* now have that varsA == varsB */
	if ((A->getStorageType() == SPARSE) && (B->getStorageType() == SPARSE)) {
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

	if ((A->getStorageType() == DENSE) && (B->getStorageType() == DENSE)) {
		_intendedStorageType = DENSE;
		return;
	}

	/* have that varsA == varsB and adding dense to sparse */
	_intendedStorageType = EVAL;	/* temporary hack */
	/* _intendedStorageType = DENSE; */
	/* TODO: convert sparse factor to dense factor */
}
<<<<<<< HEAD

FactorAdditionOp::FactorAdditionOp(drwnGeneralFactor *target,
		const vector<const drwnGeneralFactor *>& A) :
    drwnFactorNAryOp(target, A)
{	// do nothing
}

void FactorAdditionOp::execute()
{
	if (_generalFactors.size() == 2) {
		if (_intendedStorageType == EVAL) {
			return;
		} else if (_intendedStorageType == SPARSE) {
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
}
=======
>>>>>>> 5477d5d4f80732bdb8e6f07f513126e864b9fc40

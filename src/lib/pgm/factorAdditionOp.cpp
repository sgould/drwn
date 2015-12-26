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
		return;
	}

	if ((A->getStorageType() == SPARSE) && (B->getStorageType() == SPARSE)) {
		set<int> targetVars(ordTargetVars.begin(), ordTargetVars.end());	

		for (set<int>::iterator si = targetVars.begin(); si != targetVars.end(); si++) {
			vector<int>::iterator it = find(ordVarsA.begin(), ordVarsA.end(), *si);
			_targetToA.push_back(it - ordVarsA.begin());
			it = find(ordVarsB.begin(), ordVarsB.end(), *si);
			_targetToB.push_back(it - ordVarsB.begin());
		}

		return;
	}
}

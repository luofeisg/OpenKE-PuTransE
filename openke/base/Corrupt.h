#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"


 // TODO: Corrupts are used in training sampling and triple classification. While an adjustment for sampling
 // 	  is not necessary because we rely on a sampled, enumerated trainList in the ParallelUniverse configuration, 
 //  	  in triple classification we rely on the global training set. 
 //       So which approach to follow? 
 //				-- understand corrupt methods
 //             	--> return triple which are not in trainList
 //					--> but if corrupted entity is not in knowledge graph at all 
 // 					it has to be actually always false?
 //					--> on the other side: we have to rather exclude triples which have to be added as 
 // 					positive triples once. 
 //				-- if necessary changes are too complex: Write tc in python?
 //   			-- is it really necessary for tc
 //				--> Answer: If we gather entity which has been deleted from the KG in the corruption process for triple classification, 
 // 						the corrupted triple is ensured to be false. We expect the model to classify it as
 //							false anyway, so we can rely on the current version of the corrupt_[head | tail]
 //							algorithm

INT corrupt_head(INT id, INT h, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		
	// INT relation_focus = NULL;
    // if(!incrementalSetting){
    //     relation_focus = rand(0, relationTotal); //sample relation r from R
    // } else {
    //     // In case of incremental setting sample relation from currently contained relations in the train set
    //     INT rand_contained_relations_index = rand(0, num_currently_contained_relations);
    //     relation_focus = currently_contained_relations[rand_contained_relations_index]; 
    // }
		
		INT tmp = rand_max(id, entityTotal - 1);
		// INT tmp = NULL;
		// if(incrementalSetting){ 
		// 	// Check if randomly generated entity was deleted - if true: generate another entity
		// 	tmp = rand_max(id, entityTotal - 1);
		// 	} 
		// while (!checkIfEntityExists(tmp) || tmp == -1);
		
		if (tmp < h)
			return tmp;
		else
			return tmp + 1;
	}
	// What if h is not in trainList --> error! So actually have all evaluate entities be in train data
	// ANALYSE: What if h not in trainList and rigHead[h] = -1, lefHead[h] = 0
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	// INT tmp = -1;
	// do { 
	// 	// Check if randomly generated entity was deleted - if true: generate another entity
	// 	tmp = rand_max(id, entityTotal - (rr - ll + 1));
	// 	} 
	// while (!checkIfEntityExists(tmp) || tmp == -1);

	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		
		INT tmp = rand_max(id, entityTotal - 1);
		// INT tmp = -1;
		// // if (incrementalSetting) { 
		// // 	// Check if randomly generated entity was deleted - if true: generate another entity
		// // }else{
		// // 	tmp = rand_max(id, entityTotal - 1);
		// // 	} 
		// // while (!checkIfEntityExists(tmp) || tmp == -1);
		
		if (tmp < t)
			return tmp;
		else
			return tmp + 1;
	}
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	// INT tmp = -1;
	// do { 
	// 	// Check if randomly generated entity was deleted - if true: generate another entity
	// 	tmp = rand_max(id, entityTotal - (rr - ll + 1));
	// 	} 
	// while (!checkIfEntityExists(tmp) || tmp == -1);

	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel(INT id, INT h, INT t, INT r, bool p = false, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, relationTotal - 1);
		if (tmp < r)
			return tmp;
		else
			return tmp + 1;
	}
	lef = lefRel[h] - 1;
	rig = rigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefRel[h];
	rig = rigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp;
	if(p == false) {	
		tmp = rand_max(id, relationTotal - (rr - ll + 1));
	}
	else {
		INT start = r * (relationTotal - 1);
		REAL sum = 1;
		bool *record = (bool *)calloc(relationTotal - 1, sizeof(bool));
		for (INT i = ll; i <= rr; ++i){
			if (trainRel[i].r > r){
				sum -= prob[start + trainRel[i].r-1];
				record[trainRel[i].r-1] = true;
			}
			else if (trainRel[i].r < r){
				sum -= prob[start + trainRel[i].r];
				record[trainRel[i].r] = true;
			}
		}		
		REAL *prob_tmp = (REAL *)calloc(relationTotal-(rr-ll+1), sizeof(REAL));
		INT cnt = 0;
		REAL rec = 0;
		for (INT i = start; i < start + relationTotal - 1; ++i) {
			if (record[i-start])
				continue;
			rec += prob[i] / sum;
			prob_tmp[cnt++] = rec;
		}
		REAL m = rand_max(id, 10000) / 10000.0;
		lef = 0;
		rig = cnt - 1;
		while (lef < rig) {
			mid = (lef + rig) >> 1;
			if (prob_tmp[mid] < m) 
				lef = mid + 1; 
			else
				rig = mid;
		}
		tmp = rig;
		free(prob_tmp);
		free(record);
	}
	if (tmp < trainRel[ll].r) return tmp;
	if (tmp > trainRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


bool _find(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

INT corrupt(INT h, INT r){
	INT ll = tail_lef[r];
	INT rr = tail_rig[r];
	INT loop = 0;
	INT t;
	while(true) {
		t = tail_type[rand(ll, rr)];
		if (not _find(h, t, r)) {
			return t;
		} else {
			loop ++;
			if (loop >= 1000) {
				return corrupt_head(0, h, r);
			}
		} 
	}
}
#endif

#ifndef READER_H
#define READER_H

#include "Setting.h"
#include "Utilities.h"
#include "Triple.h"
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>

INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
INT *lefRel2, *rigRel2;
REAL *left_mean, *right_mean;
REAL *prob;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;
Triple *trainRel2;

INT *testLef, *testRig;
INT *validLef, *validRig;


extern "C"
void importProb(REAL temp) {
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
    printf("Current temperature:%f\n", temp);
    prob = (REAL *) calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i) {
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal - 1; ++j) {
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal - 1; ++j) {
            prob[i * (relationTotal - 1) + j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

void loadHelpers(
        Triple *&trainingList,
        Triple *&trainingListHead,
        Triple *&trainingListTail,
        Triple *&trainingListRel,
        Triple *&trainingListRel2,
        INT *&frequencyRelation,
        INT *&frequencyEntity,
        INT trainingTotal,
        INT entTotal,
        INT relTotal,
        INT *&leftIndexHead,
        INT *&rightIndexHead,
        INT *&leftIndexTail,
        INT *&rightIndexTail,
        INT *&leftIndexRelation,
        INT *&rightIndexRelation,
        INT *&leftIndexRelation2,
        INT *&rightIndexRelation2,
        REAL *&leftIndex_mean,
        REAL *&rightIndex_mean
) {
    std::sort(trainingList, trainingList + trainingTotal, Triple::cmp_head);

    callocTripleArray(trainingListHead, trainingTotal);
    callocTripleArray(trainingListTail, trainingTotal);
    callocTripleArray(trainingListRel, trainingTotal);
    callocTripleArray(trainingListRel2, trainingTotal);

    callocIntArray(frequencyEntity, entityTotal);
    callocIntArray(frequencyRelation, relationTotal);

    trainingListHead[0] = trainingListTail[0] = trainingListRel[0] = trainingListRel2[0] = trainingList[0];
    frequencyEntity[trainingList[0].t] += 1;
    frequencyEntity[trainingList[0].h] += 1;
    frequencyRelation[trainingList[0].r] += 1;

    for (INT i = 1; i < trainingTotal; i++) {
        trainingListHead[i] = trainingListTail[i] = trainingListRel[i] = trainingListRel2[i] = trainingList[i];
        frequencyEntity[trainingList[i].h]++;
        frequencyEntity[trainingList[i].t]++;
        frequencyRelation[trainList[i].r]++;
    }
    std::sort(trainingListHead, trainingListHead + trainingTotal, Triple::cmp_head);
    std::sort(trainingListTail, trainingListTail + trainingTotal, Triple::cmp_tail);
    std::sort(trainingListRel, trainingListRel + trainingTotal, Triple::cmp_rel);
    std::sort(trainingListRel2, trainingListRel2 + trainingTotal, Triple::cmp_rel2);

    callocIntArray(leftIndexHead, entityTotal);
    callocIntArray(rightIndexHead, entityTotal);
    callocIntArray(leftIndexTail, entityTotal);
    callocIntArray(rightIndexTail, entityTotal);
    callocIntArray(leftIndexRelation, entityTotal);
    callocIntArray(rightIndexRelation, entityTotal);
    callocIntArray(leftIndexRelation2, relationTotal);
    callocIntArray(rightIndexRelation2, relationTotal);

    memset(rightIndexHead, -1, sizeof(INT) * entityTotal);
    memset(rightIndexTail, -1, sizeof(INT) * entityTotal);
    memset(rightIndexRelation, -1, sizeof(INT) * entityTotal);
    memset(rightIndexRelation2, -1, sizeof(INT) * relationTotal);
    
    for (INT i = 1; i < trainingTotal; i++) {
        if (trainingListTail[i].t != trainingListTail[i - 1].t) {
            rightIndexTail[trainingListTail[i - 1].t] = i - 1;
            leftIndexTail[trainingListTail[i].t] = i;
        }
        if (trainingListHead[i].h != trainingListHead[i - 1].h) {
            rightIndexHead[trainingListHead[i - 1].h] = i - 1;
            leftIndexHead[trainingListHead[i].h] = i;
        }
        if (trainingListRel[i].h != trainingListRel[i - 1].h) {
            rightIndexRelation[trainingListRel[i - 1].h] = i - 1;
            leftIndexRelation[trainingListRel[i].h] = i;
        }
        if (trainingListRel2[i].r != trainingListRel2[i - 1].r) {
            rightIndexRelation2[trainingListRel2[i - 1].r] = i - 1;
            leftIndexRelation2[trainingListRel2[i].r] = i;
        }
    }
    
    leftIndexHead[trainingListHead[0].h] = 0;
    rightIndexHead[trainingListHead[trainingTotal - 1].h] = trainingTotal - 1;
    leftIndexTail[trainingListTail[0].t] = 0;
    rightIndexTail[trainingListTail[trainingTotal - 1].t] = trainingTotal - 1;
    leftIndexRelation[trainingListRel[0].h] = 0;
    rightIndexRelation[trainingListRel[trainingTotal - 1].h] = trainingTotal - 1;
    leftIndexRelation2[trainingListRel2[0].r] = 0;
    rightIndexRelation2[trainingListRel2[trainingTotal - 1].r] = trainingTotal - 1;
    
    callocRealArray(leftIndex_mean, relationTotal);
    callocRealArray(rightIndex_mean, relationTotal);

    for (INT i = 0; i < entTotal; i++) {
        for (INT j = leftIndexHead[i] + 1; j <= rightIndexHead[i]; j++)
            if (trainingListHead[j].r != trainingListHead[j - 1].r)
                leftIndex_mean[trainingListHead[j].r] += 1.0;
        if (leftIndexHead[i] <= rightIndexHead[i])
            leftIndex_mean[trainingListHead[leftIndexHead[i]].r] += 1.0;
        for (INT j = leftIndexTail[i] + 1; j <= rightIndexTail[i]; j++)
            if (trainingListTail[j].r != trainingListTail[j - 1].r)
                rightIndex_mean[trainingListTail[j].r] += 1.0;
        if (leftIndexTail[i] <= rightIndexTail[i])
            rightIndex_mean[trainingListTail[leftIndexTail[i]].r] += 1.0;
    }
    for (INT i = 0; i < relTotal; i++) {
        leftIndex_mean[i] = frequencyRelation[i] / leftIndex_mean[i];
        rightIndex_mean[i] = frequencyRelation[i] / rightIndex_mean[i];
    }
}

extern "C"
void importTrainFiles() {

    printf("The toolkit is importing datasets.\n");
    FILE *fin;
    int tmp;
    
    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    relationTotal = getLineNum(fin);
    // tmp = fscanf(fin, "%ld", &relationTotal);
    printf("The total of relations is %ld.\n", relationTotal);
    fclose(fin);

    fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    entityTotal = getLineNum(fin);
    // tmp = fscanf(fin, "%ld", &entityTotal);
    printf("The total of entities is %ld.\n", entityTotal);
    fclose(fin);

    fin = fopen((inPath + "train2id.txt").c_str(), "r");
    trainTotal = getLineNum(fin);
    // tmp = fscanf(fin, "%ld", &trainTotal);
    trainList = (Triple *) calloc(trainTotal, sizeof(Triple));
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(fin, "%ld", &trainList[i].h);
        tmp = fscanf(fin, "%ld", &trainList[i].t);
        tmp = fscanf(fin, "%ld", &trainList[i].r);
    }
    fclose(fin);

    std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
    tmp = trainTotal;
    trainTotal = 1;
    for (INT i = 1; i < tmp; i++)
        if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r ||
            trainList[i].t != trainList[i - 1].t) {
            trainList[trainTotal] = trainList[i];
            trainTotal++;
        }
    printf("The total of train triples is %ld.\n", trainTotal);

    loadHelpers(
            trainList,
            trainHead,
            trainTail,
            trainRel,
            trainRel2,
            freqRel,
            freqEnt,
            trainTotal,
            entityTotal,
            relationTotal,
            lefHead,
            rigHead,
            lefTail,
            rigTail,
            lefRel,
            rigRel,
            lefRel2,
            rigRel2,
            left_mean,
            right_mean
    );


}

Triple *testList;
Triple *validList;
Triple *tripleList;

bool loadAllTriple = false;
extern "C"
void activateLoadOfAllTriples(bool){
    loadAllTriple = true;
}

extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;

    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    // tmp = fscanf(fin, "%ld", &relationTotal);
    relationTotal = getLineNum(fin);
    fclose(fin);

    fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    // tmp = fscanf(fin, "%ld", &entityTotal);
    entityTotal = getLineNum(fin);
    fclose(fin);

    FILE *f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    FILE *f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    FILE *f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    // tmp = fscanf(f_kb1, "%ld", &testTotal);
    // tmp = fscanf(f_kb2, "%ld", &trainTotal);
    // tmp = fscanf(f_kb3, "%ld", &validTotal);
    testTotal = getLineNum(f_kb1);
    trainTotal = getLineNum(f_kb2);
    validTotal = getLineNum(f_kb3);
    tripleTotal = testTotal + trainTotal + validTotal;
    
    testList = (Triple *) calloc(testTotal, sizeof(Triple));
    validList = (Triple *) calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *) calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    if (loadAllTriple){
        FILE *fin = fopen((inPath + "global_triple2id.txt").c_str(), "r");
        tripleTotal = getLineNum(fin);
        callocTripleArray(tripleList ,tripleTotal);
        printf("Captured %ld triples in total.\n", tripleTotal);
        
        for (INT i = 0; i < tripleTotal; i++) {
            tmp = fscanf(fin, "%ld", &tripleList[i].h);
            tmp = fscanf(fin, "%ld", &tripleList[i].t);
            tmp = fscanf(fin, "%ld", &tripleList[i].r);
        }
        fclose(fin);
    }

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);
    printf("The total of train triples is %ld.\n", trainTotal);

    testLef = (INT *) calloc(relationTotal, sizeof(INT));
    testRig = (INT *) calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
        if (testList[i].r != testList[i - 1].r) {
            testRig[testList[i - 1].r] = i - 1;
            testLef[testList[i].r] = i;
        }
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *) calloc(relationTotal, sizeof(INT));
    validRig = (INT *) calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT) * relationTotal);
    memset(validRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < validTotal; i++) {
        if (validList[i].r != validList[i - 1].r) {
            validRig[validList[i - 1].r] = i - 1;
            validLef[validList[i].r] = i;
        }
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

INT *head_lef;
INT *head_rig;
INT *tail_lef;
INT *tail_rig;
INT *head_type;
INT *tail_type;

extern "C"
void importTypeFiles() {

    head_lef = (INT *) calloc(relationTotal, sizeof(INT));
    head_rig = (INT *) calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *) calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *) calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    FILE *f_type = fopen((inPath + "type_constrain.txt").c_str(), "r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *) calloc(total_lef, sizeof(INT));
    tail_type = (INT *) calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "type_constrain.txt").c_str(), "r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}

void resetHelpers() {
    resetTripleHelper(trainList);
    resetTripleHelper(trainHead);
    resetTripleHelper(trainTail);
    resetTripleHelper(trainRel);
    resetTripleHelper(trainRel2);
    resetIntHelper(freqEnt);
    resetIntHelper(freqRel);
    resetIntHelper(lefHead);
    resetIntHelper(rigHead);
    resetIntHelper(lefTail);
    resetIntHelper(rigTail);
    resetIntHelper(lefRel);
    resetIntHelper(rigRel);
    resetIntHelper(lefRel2);
    resetIntHelper(rigRel2);
    resetRealHelper(left_mean);
    resetRealHelper(right_mean);
}

#endif

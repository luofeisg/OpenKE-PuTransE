#ifndef CHECKS_H
#define CHECKS_H

#include "Triple.h"
#include "Utilities.h"
#include "Reader.h"
#include "UniverseSetting.h"
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>


void checkHelper(INT entOrRel, INT *&indexArray, char const *indexArrayName) {
    if ((indexArray[entOrRel] >= 0) && (indexArray[entOrRel] < trainTotal)) {
        // Check index of first occurring entity or relation in indexing Arrays
        if (((isEqual(indexArrayName, "lefHead") || isEqual(indexArrayName, "lefRel")) &&
             trainHead[0].h == entOrRel) ||
            (isEqual(indexArrayName, "lefTail") && trainTail[0].t == entOrRel) ||
            (isEqual(indexArrayName, "lefRel2") && trainRel2[0].r == entOrRel)) {
            if (indexArray[entOrRel] != 0) {
                printf("False index entry in %s for lowest entity or relation id %ld.\n", indexArrayName, entOrRel);
                exit(EXIT_FAILURE);
            }
        }

        if (indexArray[entOrRel] > 0) {
            if (isEqual(indexArrayName, "lefHead") || isEqual(indexArrayName, "rigHead") ||
                isEqual(indexArrayName, "lefRel") ||
                isEqual(indexArrayName, "rigRel")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainHead[indexArray[entOrRel]].h);
            } else if (isEqual(indexArrayName, "lefTail") || isEqual(indexArrayName, "rigTail")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainTail[indexArray[entOrRel]].t);
            } else if (isEqual(indexArrayName, "lefRel2") || isEqual(indexArrayName, "rigRel2")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainRel2[indexArray[entOrRel]].r);
            }
        }

    } else if (isEqual(indexArrayName, "rigHead") || isEqual(indexArrayName, "rigTail") ||
               isEqual(indexArrayName, "rigRel") || isEqual(indexArrayName, "rigRel2")) {
        if (indexArray[entOrRel] != -1) {
            printf("Mapped index in %s of entity %ld out of bound.\n", indexArrayName, entOrRel);
            exit(EXIT_FAILURE);
        }

    } else {
        printf("Mapped index in %s of entity %ld out of bound.\n", indexArrayName, entOrRel);
        exit(EXIT_FAILURE);
    }
}

void checkHelpers(INT entTotal, INT relTotal) {
    printf("Verify index entries in Helpers.\n");
    for (INT i = 0; i < entTotal; i++) {
        checkHelper(i, lefHead, "lefHead");
        checkHelper(i, rigHead, "rigHead");
        checkHelper(i, lefTail, "lefTail");
        checkHelper(i, rigTail, "rigTail");
        checkHelper(i, lefRel, "lefRel");
        checkHelper(i, rigRel, "rigRel");

        //check entity frequencies
        int entityFreq = 0;
        for (int j = 0; j < trainTotal; j++) {
            if (trainList[j].h == i)
                entityFreq++;

            if (trainList[j].t == i)
                entityFreq++;
        }

        if (entityFreq != freqEnt[i])
            printf("false frequency for entity %ld.\n", i);
    }

    for (INT i = 0; i < relTotal; i++) {
        checkHelper(i, lefRel2, "lefRel2");
        checkHelper(i, rigRel2, "rigRel2");

        int relationFreq = 0;
        for (INT j = 0; j < trainTotal; j++) {
            if (trainList[j].r == i)
                relationFreq++;
        }

        if (relationFreq != freqRel[i])
            printf("false frequency for relation %ld.\n", i);
    }
    printf("Helpers checked successfully.\n");
}

void checkSampling(
    INT *batch_h,
    INT *batch_t,
    INT *batch_r,
    REAL *batch_y,
    Triple *trainingList,
    INT trainingTotal,
    INT batchSize) {

    INT h, r, t;
    INT y;
    bool found;

    for(INT i=0;i<batchSize;i++){
        if (swap){
            h = entity_remapping[batch_h[i]];
            r = relation_remapping[batch_r[i]];
            t = entity_remapping[batch_t[i]];
            y = batch_y[i];
        } else {
            h = batch_h[i];
            r = batch_r[i];
            t = batch_t[i];
            y = batch_y[i];
        }
        found = false;
        
        for(INT j=0;j<trainingTotal;j++){
            if(trainingList[j].h == h && trainingList[j].r == r && trainingList[j].t == t){
                found = true;
                break;
            }
        }

        if(!found && y == 1){
            printf("Triple (%ld, %ld, %ld) not found in Universe or train list.\n", h, r, t);
            exit(EXIT_FAILURE);
        }else if(found && y == -1){
            printf("Negative triple (%ld, %ld, %ld) has been found in Universe or train list.\n", h, r, t);
            exit(EXIT_FAILURE);
        }
    }
    printf("Batch is valid.\n");
}


#endif

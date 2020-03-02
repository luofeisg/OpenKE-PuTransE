#ifndef CHECKS_H
#define CHECKS_H

#include "Triple.h"
#include "Reader.h"
#include "UniverseSetting.h"
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>

void checkUniverseTrainingTriples() {
    INT num_of_no_entry = 0;
    bool no_entry;
    for (int i = 0; i < trainTotalUniverse; i++) {

        no_entry = true;
        INT head = trainListUniverse[i].h;
        INT rel = trainListUniverse[i].r;
        INT tail = trainListUniverse[i].t;

        for (int j = 0; j < trainTotal; j++) {
            if (head == trainList[j].h && rel == trainList[j].r && tail == trainList[j].t) {
                no_entry = false;
                break;
            }
        }
        if (no_entry == true) {
            num_of_no_entry++;
            no_entry = false;
            printf("Unknown triple: (%ld, %ld, %ld).\n", head, rel, tail);
        }
    }
    printf("Number of unknown triples: %ld.\n", num_of_no_entry);
}

void checkEnumeration() {
    INT num_of_false_entry = 0;
    INT max_entity = 0;
    INT max_relation = 0;

    std::set<INT> entities;
    std::set<INT> relations;

    for (int i = 0; i < trainTotalUniverse; i++) {
        entities.insert(trainListUniverse[i].h);
        entities.insert(trainListUniverse[i].t);
        relations.insert(trainListUniverse[i].r);
        if (trainListUniverse[i].h != entity_remapping[trainListUniverseEnum[i].h] ||
            trainListUniverse[i].r != relation_remapping[trainListUniverseEnum[i].r] ||
            trainListUniverse[i].t != entity_remapping[trainListUniverseEnum[i].t]) {
            printf("Real triple: (%ld, %ld, %ld).\n", trainListUniverse[i].h, trainListUniverse[i].r,
                   trainListUniverse[i].t);
            printf("Mapped triple: (%ld, %ld, %ld).\n", trainListUniverseEnum[i].h, trainListUniverseEnum[i].r,
                   trainListUniverseEnum[i].t);
            num_of_false_entry++;
        }

        if (trainListUniverseEnum[i].h > max_entity)
            max_entity = trainListUniverseEnum[i].h;

        if (trainListUniverseEnum[i].t > max_entity)
            max_entity = trainListUniverseEnum[i].t;

        if (trainListUniverseEnum[i].r > max_relation)
            max_relation = trainListUniverseEnum[i].r;
    }

    printf("-----Check enumeration--------\n");
    printf("Number of falsely mapped triples: %ld.\n", num_of_false_entry);
    printf("-------------\n");
    printf("Number of entities: %ld.\n", entities.size());
    printf("Number of relations: %ld.\n", relations.size());
    printf("-------------\n");
    printf("entityTotalUniverse: %ld VS max_entity_id: %ld.\n", entityTotalUniverse, max_entity);
    printf("relationTotalUniverse: %ld VS max_relation_id: %ld.\n", relationTotalUniverse, max_relation);
    printf("-------------\n");
}

void verifyIndexEntry(INT entity, char const *indexArrayName, INT tripleArrayEntry) {
    if (entity != tripleArrayEntry) {
        printf("Entity %ld falsely mapped by %s.\n", entity, indexArrayName);
        exit(EXIT_FAILURE);
    }
}

bool isEqual(const char *string1, const char *string2) {
    return strcmp(string1, string2) == 0;
}

void checkHelper(INT entOrRel, INT *&indexArray, char const *indexArrayName) {
    if ((indexArray[entOrRel] >= 0) && (indexArray[entOrRel] < trainTotalUniverse)) {
        // Check index of first occurring entity or relation in indexing Arrays
        if (((isEqual(indexArrayName, "lefHead") || isEqual(indexArrayName, "lefRel")) &&
             trainHeadUniverse[0].h == entOrRel) ||
            (isEqual(indexArrayName, "lefTail") && trainTailUniverse[0].t == entOrRel) ||
            (isEqual(indexArrayName, "lefRel2") && trainRel2Universe[0].r == entOrRel)) {
            if (indexArray[entOrRel] != 0) {
                printf("False index entry in %s for lowest entity or relation id %ld.\n", indexArrayName, entOrRel);
                exit(EXIT_FAILURE);
            }
        }

        if (indexArray[entOrRel] > 0) {
            if (isEqual(indexArrayName, "lefHead") || isEqual(indexArrayName, "rigHead") ||
                isEqual(indexArrayName, "lefRel") ||
                isEqual(indexArrayName, "rigRel")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainHeadUniverse[indexArray[entOrRel]].h);
            } else if (isEqual(indexArrayName, "lefTail") || isEqual(indexArrayName, "rigTail")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainTailUniverse[indexArray[entOrRel]].t);
            } else if (isEqual(indexArrayName, "lefRel2") || isEqual(indexArrayName, "rigRel2")) {
                verifyIndexEntry(entOrRel, indexArrayName, trainRel2Universe[indexArray[entOrRel]].r);
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

void checkUniverseHelpers() {
    printf("Verify index entries in Helpers.\n");
    for (INT i = 0; i < entityTotalUniverse; i++) {
        checkHelper(i, lefHeadUniverse, "lefHead");
        checkHelper(i, rigHeadUniverse, "rigHead");
        checkHelper(i, lefTailUniverse, "lefTail");
        checkHelper(i, rigTailUniverse, "rigTail");
        checkHelper(i, lefRelUniverse, "lefRel");
        checkHelper(i, rigRelUniverse, "rigRel");

        //check entity frequencies
        int entityFreq = 0;
        for (int j = 0; j < trainTotalUniverse; j++) {
            if (trainListUniverseEnum[j].h == i)
                entityFreq++;

            if (trainListUniverseEnum[j].t == i)
                entityFreq++;
        }

        if (entityFreq != freqEntUniverse[i])
            printf("false frequency for entity %ld.\n", i);
    }

    for (INT i = 0; i < relationTotalUniverse; i++) {
        checkHelper(i, lefRel2Universe, "lefRel2");
        checkHelper(i, rigRel2Universe, "rigRel2");

        int relationFreq = 0;
        for (INT j = 0; j < trainTotalUniverse; j++) {
            if (trainListUniverseEnum[j].r == i)
                relationFreq++;
        }

        if (relationFreq != freqRelUniverse[i])
            printf("false frequency for relation %ld.\n", i);
    }
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
        h = entity_remapping[batch_h[i]];
        r = relation_remapping[batch_r[i]];
        t = entity_remapping[batch_t[i]];
        y = batch_y[i];

        found = false;
        
        for(INT j=0;j<trainingTotal;j++){
            if(trainingList[j].h == h && trainingList[j].r == r && trainingList[j].t == t){
                found = true;
                break;
            }
        }

        if(!found && y == 1){
            printf("Triple (%ld, %ld, %ld) not found in Universe.\n", h, r, t);
            exit(EXIT_FAILURE);
        }else if(found && y == -1){
            printf("Negative triple (%ld, %ld, %ld) has been found in Universe.\n", h, r, t);
            exit(EXIT_FAILURE);
        }
    }
    printf("Batch is valid.\n");
}


#endif

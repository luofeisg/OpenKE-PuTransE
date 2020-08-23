/*
===================== Procedure for loading data incrementally =========================
*/
// 1) We initialize our incremental Dataloader by setting the following variables



#ifndef INCREMENTAL_H
#define INCREMENTAL_H


#include "Setting.h"
#include "Utilities.h"
#include "Triple.h"
#include "Reader.h"
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>

/*
===================== Variables for incremental loading of triple operations =========================
*/

bool incrementalSetting = false;
extern "C"
void activateIncrementalSetting() {
	incrementalSetting = true;
}

INT num_snapshots = 0;

extern "C"
void setNumSnapshots(INT num) {
	num_snapshots = num;
}

extern "C"
INT getNumSnapshots() {
	return num_snapshots;
}


// In the incremental setting the variable entityTotal adapts the number of entities which has been
// added to the graph so far, no matter if they have been deleted in the meantime. By doing so, it 
// is assured that the written code for the incremental setting is consistent with the original
// version of the OpenKE framework.
// all_entities = num_currently_contained_entities + num_deleted_entities

// static std::set<INT> all_entities;
// static std::set<INT> currently_contained_entities;
// static std::set<INT> deleted_entities;
// std::set<INT> all_relations;
// std::set<INT> currently_contained_relations;
// std::set<INT> deleted_relations;

// Since trainList is loaded in TrainDataLoader and tripleList in TestDataloader we can rely 
// on the same entity sets and variables which support the processing of triple operations
// into the specific triple lists trainList and tripleList 

// Variables for entire triple operation set
INT *all_entities;
INT num_all_entities = 0;
INT *currently_contained_entities;
INT num_currently_contained_entities = 0;
INT *deleted_entities;
INT num_deleted_entities = 0;


INT *all_relations;
INT num_all_relations = 0;
INT *currently_contained_relations;
INT num_currently_contained_relations = 0;
INT *deleted_relations;
INT num_deleted_relations = 0;


INT next_operation_id = 0;
INT totalOperations = 0;
bool lastOperationFinished = false;

INT maxEntity = 0;
INT maxRelation = 0;

TripleOperation *KnowledgeGraphOperations;

// Variables for train operation set
INT *all_train_entities;
INT num_all_train_entities = 0;
INT *currently_contained_train_entities;
INT num_currently_contained_train_entities = 0;
INT *deleted_train_entities;
INT num_deleted_train_entities = 0;


INT *all_train_relations;
INT num_all_train_relations = 0;
INT *currently_contained_train_relations;
INT num_currently_contained_train_relations = 0;
INT *deleted_train_relations;
INT num_deleted_train_relations = 0;


INT next_train_operation_id = 0;
INT totaltrain_Operations = 0;
bool lastTrainOperationFinished = false;

INT maxTrainEntity = 0;
INT maxTrainRelation = 0;

TripleOperation *KnowledgeGraphTrainOperations;



extern "C"
INT getNumCurrentlyContainedEntities() {
	// return currently_contained_entities.size();
    return num_currently_contained_entities;
}

int numOperationsRate = 0;
extern "C"
void setNumOperationsRate(INT num) {
	numOperationsRate = num;
}

int getNumOperationsRate(INT num) {
	return numOperationsRate;
}

void allocTripleArray(Triple* &arr, INT length) {
    if(arr == NULL)
        arr = (Triple *) calloc(length, sizeof(Triple));
        
    if (!arr) {
        printf("out of mem\n");
        exit(EXIT_FAILURE);
    }

    if(arr != NULL)
        arr = (Triple *) realloc(arr, length * sizeof(Triple));
}

void createArrayEntry(INT *&arr, INT &arrayLength, INT entry) {
	arrayLength++;
    callocIntArray(arr, arrayLength);
    arr[arrayLength-1] = entry;
}

void deleteArrayEntry(INT *&arr, INT &arrayLength, INT entry) {	
    for(int i=0; i<arrayLength; i++){
        if(arr[i] == entry){
            for(int j=i; j<arrayLength-1; j++){
                arr[j] = arr[j+1];
            }
            arrayLength--;
            callocIntArray(arr, arrayLength);
            break;
        }
    }
}

extern "C"
void readGlobalNumEntities() {
	// Load entityTotal
    printf("Get total number of entities occuring during all snapshots.\n");

    FILE *fin;
    int tmp;
    fin = fopen((inPath + "incremental/" + "/entity2id.txt").c_str(), "r");
    entityTotal = getLineNum(fin);
    printf("Folder: %s.\n", (inPath + "incremental/" + "/entity2id.txt").c_str());
    fclose(fin);
}

extern "C"
void readGlobalNumRelations() {
    printf("Get total number of relations occuring during all snapshots.\n");

    FILE *fin;
    int tmp;
    fin = fopen((inPath + "incremental/" + "/relation2id.txt").c_str(), "r");
    printf("Folder: %s.\n", (inPath + "incremental/" + "/relation2id.txt").c_str());
    relationTotal = getLineNum(fin);
    fclose(fin);
}

extern "C"
void initializeIncrementalSetting() {
    printf("Initialize knowledge graph.\n");
    incrementalSetting = true;

    FILE *fin;
    int tmp;
    
    readGlobalNumRelations();
    readGlobalNumEntities();
}

void resetSnapShot() {
    // Called in the beginning of each snapshot
    // Used for both training and triple operations (tripleList)
    if (KnowledgeGraphOperations != NULL) {
        free(KnowledgeGraphOperations);
        KnowledgeGraphOperations = NULL;
        totalOperations = 0;
        next_operation_id = 0;
        numOperationsRate = 0;
        lastOperationFinished = false;
    }
}

void resetTestData() {
    // Called in the beginning of each snapshot
    if (testList != NULL) {
        free(testList);
        testList = NULL;
    }
}

void resetValidData() {
    // Called in the beginning of each snapshot
    if (validList != NULL) {
        free(validList);
        validList = NULL;
    }
}

extern "C"
void loadTestData(int snapshot) {
    resetTestData();
    printf("Import test data for snapshot: %d.\n", snapshot);
    
    std::string snapshot_folder = int_to_string(snapshot);
    
    FILE *fin;
    int tmp;

    fin = fopen((inPath + "incremental/" + snapshot_folder + "/test2id.txt").c_str(), "r");
    printf("Folder: %s.\n", (inPath + "incremental/" + snapshot_folder + "/test2id.txt").c_str());
    
    testTotal = getLineNum(fin);
    testList = (Triple *) calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(fin, "%ld", &testList[i].h);
        tmp = fscanf(fin, "%ld", &testList[i].t);
        tmp = fscanf(fin, "%ld", &testList[i].r);
    }
    fclose(fin);

    printf("Number of test triples in snapshot: %ld", testTotal);
}

extern "C"
void loadValidData(int snapshot) {
    resetValidData();
    printf("Import test data for snapshot: %d.\n", snapshot);
    
    std::string snapshot_folder = int_to_string(snapshot);
    
    FILE *fin;
    int tmp;

    fin = fopen((inPath + "incremental/" + snapshot_folder + "/valid2id.txt").c_str(), "r");
    printf("Folder: %s.\n", (inPath + "incremental/" + snapshot_folder + "/valid2id.txt").c_str());
    
    validTotal = getLineNum(fin);
    validList = (Triple *) calloc(validTotal, sizeof(Triple));
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(fin, "%ld", &validList[i].h);
        tmp = fscanf(fin, "%ld", &validList[i].t);
        tmp = fscanf(fin, "%ld", &validList[i].r);
    }
    fclose(fin);

    printf("Number of valid triples in snapshot: %ld", validTotal);   
}

// Read out all training operations
extern "C"
void initializeTrainingOperations(int snapshot) {
    resetSnapShot();
    printf("Import snapshot: %d.\n", snapshot);
    
    std::string snapshot_folder = int_to_string(snapshot);
    
    FILE *fin;
    int tmp;

    fin = fopen((inPath + "incremental/" + snapshot_folder + "/train-op2id.txt").c_str(), "r");
    printf("Folder: %s.\n", (inPath + "incremental/" + snapshot_folder + "/train-op2id.txt").c_str());
    
    totalOperations = getLineNum(fin);
    KnowledgeGraphOperations = (TripleOperation *) calloc(totalOperations, sizeof(TripleOperation));
    for (INT i = 0; i < totalOperations; i++) {
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.h);
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.t);
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.r);
        tmp = fscanf(fin, "%s", &KnowledgeGraphOperations[i].operation);
    }
    fclose(fin);
    
}

// Read out all triple operations
extern "C"
void initializeTripleOperations(int snapshot) {
    resetSnapShot();
    printf("Import triple operations for snapshot: %d.\n", snapshot);
    
    std::string snapshot_folder = int_to_string(snapshot);
    
    FILE *fin;
    int tmp;

    fin = fopen((inPath + "incremental/" + snapshot_folder + "/triple-op2id.txt").c_str(), "r");
    printf("Folder: %s.\n", (inPath + "incremental/" + snapshot_folder + "/triple-op2id.txt").c_str());
    
    totalOperations = getLineNum(fin);
    printf("Captured %ld triple operations in snapshot %d.\n", totalOperations, snapshot);
    KnowledgeGraphOperations = (TripleOperation *) calloc(totalOperations, sizeof(TripleOperation));
    for (INT i = 0; i < totalOperations; i++) {
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.h);
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.t);
        tmp = fscanf(fin, "%ld", &KnowledgeGraphOperations[i].triple.r);
        tmp = fscanf(fin, "%s", &KnowledgeGraphOperations[i].operation);
    }
    fclose(fin);
    printf("Finished loading KG operations.\n");
}

// Checks if entity ent exists in tripleList
bool checkIfEntityExists(INT ent){
    bool entityExists = false;
    for(int i=0; i<tripleTotal; i++){
        if(tripleList[i].h == ent || tripleList[i].t == ent){
            entityExists = true;
            break;
        }
    }    
    return entityExists;
}

bool checkIfTrainEntityExists(INT ent){
    bool entityExists = false;
    for(int i=0; i<trainTotal; i++){
        if(trainList[i].h == ent || trainList[i].t == ent){
            entityExists = true;
            break;
        }
    }    
    return entityExists;
}

// Checks if relation rel exists in tripleList
bool checkIfRelationExists(INT rel){
    bool relationExists = false;
    for(int i=0; i<tripleTotal; i++){
        if(tripleList[i].r == rel){
            relationExists = true;
            break;
        }
    }    
    return relationExists;
}

// Checks if relation rel exists in tripleList
bool checkIfTrainRelationExists(INT rel){
    bool relationExists = false;
    for(int i=0; i<trainTotal; i++){
        if(trainList[i].r == rel){
            relationExists = true;
            break;
        }
    }    
    return relationExists;
}

bool checkIfEntityIsNew(INT entity){
    bool entityIsNew = true;
    for(int i=0; i<num_all_entities; i++){
        if(all_entities[i] == entity){
            entityIsNew = false;
            break;
        }
    }
    return entityIsNew;
}

bool checkIfTrainEntityIsNew(INT entity){
    bool entityIsNew = true;
    for(int i=0; i<num_all_train_entities; i++){
        if(all_train_entities[i] == entity){
            entityIsNew = false;
            break;
        }
    }
    return entityIsNew;
}

bool checkIfRelationIsNew(INT rel){
    bool relationIsNew = true;
    for(int i=0; i<num_all_relations; i++){
        if(all_relations[i] == rel){
            relationIsNew = false;
            break;
        }
    }
    return relationIsNew;
}

bool checkIfTrainRelationsIsNew(INT rel){
    bool relationIsNew = true;
    for(int i=0; i<num_all_train_relations; i++){
        if(all_train_relations[i] == rel){
            relationIsNew = false;
            break;
        }
    }
    return relationIsNew;
}

bool checkIfEntityDeleted(INT entity){
    // return deleted_entities.find(entity) != deleted_entities.end();
    bool entityDeleted = false;
    for(int i=0; i<num_deleted_entities; i++){
        if(deleted_entities[i] == entity){
            entityDeleted = true;
            break;
        }
    }
    return entityDeleted;
}

bool checkIfTrainEntityDeleted(INT entity){
    // return deleted_entities.find(entity) != deleted_entities.end();
    bool entityDeleted = false;
    for(int i=0; i<num_deleted_train_entities; i++){
        if(deleted_train_entities[i] == entity){
            entityDeleted = true;
            break;
        }
    }
    return entityDeleted;
}

void adjustEntitySet(INT ent){
    // If entity exists in KG return else entity is either new or was deleted before
    if(checkIfEntityExists(ent)){
        return;
    } else {
        if(checkIfEntityIsNew(ent)){
            createArrayEntry(all_entities, num_all_entities, ent);
        }else{
            deleteArrayEntry(deleted_entities, num_deleted_entities, ent);
        }   
        createArrayEntry(currently_contained_entities, num_currently_contained_entities, ent); 
    }
}

void adjustTrainEntitySet(INT ent){
    // If entity exists in KG return else entity is either new or was deleted before
    if(checkIfTrainEntityExists(ent)){
        return;
    } else {
        if(checkIfTrainEntityIsNew(ent)){
            createArrayEntry(all_train_entities, num_all_train_entities, ent);
        }else{
            deleteArrayEntry(deleted_train_entities, num_deleted_train_entities, ent);
        }   
        createArrayEntry(currently_contained_train_entities, num_currently_contained_train_entities, ent);
    }
}

bool checkIfRelationDeleted(INT relation){
    // return deleted_relations.find(relation) != deleted_relations.end();
        // return deleted_entities.find(entity) != deleted_entities.end();
    bool relationDeleted = false;
    for(int i=0; i<num_deleted_relations; i++){
        if(deleted_relations[i] == relation){
            relationDeleted = true;
            break;
        }
    }
    return relationDeleted;
}

bool checkIfTrainRelationDeleted(INT relation){
    bool relationDeleted = false;
    for(int i=0; i<num_deleted_train_relations; i++){
        if(deleted_train_relations[i] == relation){
            relationDeleted = true;
            break;
        }
    }
    return relationDeleted;
}

void adjustRelationSet(INT rel){
    // If relation exists in KG return else relation is either new or was deleted before
    if(checkIfRelationExists(rel)){
        return;
    } else {
        if(checkIfRelationIsNew(rel)){
            createArrayEntry(all_relations, num_all_relations, rel);
        }else{
            deleteArrayEntry(deleted_relations, num_deleted_relations, rel);
        }   
        createArrayEntry(currently_contained_relations, num_currently_contained_relations, rel);
    }
}

void adjustTrainRelationSet(INT rel){
    // If relation exists in KG return else relation is either new or was deleted before
    if(checkIfTrainRelationExists(rel)){
        return;
    } else {
        if(checkIfTrainRelationsIsNew(rel)){
            createArrayEntry(all_train_relations, num_all_train_relations, rel);
        }else{
            deleteArrayEntry(deleted_train_relations, num_deleted_train_relations, rel);
        }   
        createArrayEntry(currently_contained_train_relations, num_currently_contained_train_relations, rel);
    }
}

void insertTriple(Triple trip) {
    adjustEntitySet(trip.h);
    adjustEntitySet(trip.t);
    adjustRelationSet(trip.r);
    
    tripleTotal++;
    callocTripleArray(tripleList, tripleTotal);
    
    tripleList[tripleTotal-1].h = trip.h;
    tripleList[tripleTotal-1].t = trip.t; 
    tripleList[tripleTotal-1].r = trip.r;
}

void insertTrainTriple(Triple trip) {
    adjustTrainEntitySet(trip.h);
    adjustTrainEntitySet(trip.t);
    adjustTrainRelationSet(trip.r);
    
    trainTotal++;
    callocTripleArray(trainList, trainTotal);
    
    trainList[trainTotal-1].h = trip.h;
    trainList[trainTotal-1].t = trip.t; 
    trainList[trainTotal-1].r = trip.r;
}

void entityRemovalCheck(INT ent){
    // If entity ent does not exists after delete operation remove it from currently_contained_entities
    // and add it do deleted entities
    bool entityExists = checkIfEntityExists(ent);
    if(!entityExists){
        // currently_contained_entities.erase(ent);
        deleteArrayEntry(currently_contained_entities, num_currently_contained_entities, ent);
        // deleted_entities.insert(ent);
        createArrayEntry(deleted_entities, num_deleted_entities, ent);
    }   
}

void trainEntityRemovalCheck(INT ent){
    // If entity ent does not exists after delete operation remove it from currently_contained_entities
        // and add it do deleted entities
    bool entityExists = checkIfTrainEntityExists(ent);
    if(!entityExists){
        // currently_contained_entities.erase(ent);
        deleteArrayEntry(currently_contained_train_entities, num_currently_contained_train_entities, ent);
        // deleted_entities.insert(ent);
        createArrayEntry(deleted_train_entities, num_deleted_train_entities, ent);
    }   
}

void relationRemovalCheck(INT rel){
    bool relationExists = checkIfRelationExists(rel); 
    if(!relationExists){
        // currently_contained_relations.erase(relation);
        deleteArrayEntry(currently_contained_relations, num_currently_contained_relations, rel);
        // deleted_relations.insert(relation);
        createArrayEntry(deleted_relations, num_deleted_relations, rel);
    }
}

void TrainRelationRemovalCheck(INT rel){
    bool relationExists = checkIfTrainRelationExists(rel); 
    if(relationExists == false){
        // currently_contained_relations.erase(relation);
        deleteArrayEntry(currently_contained_train_relations, num_currently_contained_train_relations, rel);
        // deleted_relations.insert(relation);
        createArrayEntry(deleted_train_relations, num_deleted_train_relations, rel);
    }
}

void deleteTriple(Triple trip) {
    bool triple_existed = false;
    for(int i=0; i<tripleTotal; i++){
        if(tripleList[i].h == trip.h && tripleList[i].r == trip.r && tripleList[i].t == trip.t){
            triple_existed = true;
            for(int j=i; j<tripleTotal-1; j++){
                tripleList[j].h = tripleList[j+1].h;
                tripleList[j].r = tripleList[j+1].r;
                tripleList[j].t = tripleList[j+1].t;
            }
            break;
        }
    }
    
    if(triple_existed){
        tripleTotal--;
        callocTripleArray(tripleList, tripleTotal);

        entityRemovalCheck(trip.h);
        entityRemovalCheck(trip.t);
        relationRemovalCheck(trip.r);
    }else{
        printf("Triple %ld,%ld,%ld not found in KG.", trip.h, trip.t, trip.r);
        return;
    }
}

void deleteTrainTriple(Triple trip) {
    bool triple_existed = false;
    for(int i=0; i<trainTotal; i++){
        if(trainList[i].h == trip.h && trainList[i].r == trip.r && trainList[i].t == trip.t){
            triple_existed = true;
            for(int j=i; j<trainTotal-1; j++){
                trainList[j].h = trainList[j+1].h;
                trainList[j].r = trainList[j+1].r;
                trainList[j].t = trainList[j+1].t;
            }
            break;
        }
    }
    
    if(triple_existed){
        trainTotal--;
        callocTripleArray(trainList, trainTotal);

        trainEntityRemovalCheck(trip.h);
        trainEntityRemovalCheck(trip.t);
        TrainRelationRemovalCheck(trip.r);
    }else{
        printf("Train Triple %ld,%ld,%ld not found in KG.", trip.h, trip.t, trip.r);
        return;
    }
}

void resetIncrementalHelpers() {
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

void loadIncrementalHelpers(
        Triple *&trainingList,
        Triple *&trainingListHead,
        Triple *&trainingListTail,
        Triple *&trainingListRel,
        Triple *&trainingListRel2,
        INT *&frequencyRelation,
        INT *&frequencyEntity,
        INT trainTotal,
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
    std::sort(trainingList, trainingList + trainTotal, Triple::cmp_head);

    callocTripleArray(trainingListHead, trainTotal);
    callocTripleArray(trainingListTail, trainTotal);
    callocTripleArray(trainingListRel, trainTotal);
    callocTripleArray(trainingListRel2, trainTotal);

    callocIntArray(frequencyEntity, entTotal);
    callocIntArray(frequencyRelation, relTotal);

    trainingListHead[0] = trainingListTail[0] = trainingListRel[0] = trainingListRel2[0] = trainingList[0];
    frequencyEntity[trainingList[0].t] += 1;
    frequencyEntity[trainingList[0].h] += 1;
    frequencyRelation[trainingList[0].r] += 1;

    for (INT i = 1; i < trainTotal; i++) {
        trainingListHead[i] = trainingListTail[i] = trainingListRel[i] = trainingListRel2[i] = trainingList[i];
        frequencyEntity[trainingList[i].h]++;
        frequencyEntity[trainingList[i].t]++;
        frequencyRelation[trainList[i].r]++;
    }
    std::sort(trainingListHead, trainingListHead + trainTotal, Triple::cmp_head);
    std::sort(trainingListTail, trainingListTail + trainTotal, Triple::cmp_tail);
    std::sort(trainingListRel, trainingListRel + trainTotal, Triple::cmp_rel);
    std::sort(trainingListRel2, trainingListRel2 + trainTotal, Triple::cmp_rel2);

    callocIntArray(leftIndexHead, entTotal);
    callocIntArray(rightIndexHead, entTotal);
    callocIntArray(leftIndexTail, entTotal);
    callocIntArray(rightIndexTail, entTotal);
    callocIntArray(leftIndexRelation, entTotal);
    callocIntArray(rightIndexRelation, entTotal);
    callocIntArray(leftIndexRelation2, relTotal);
    callocIntArray(rightIndexRelation2, relTotal);

    memset(rightIndexHead, -1, sizeof(INT) * entTotal);
    memset(rightIndexTail, -1, sizeof(INT) * entTotal);
    memset(rightIndexRelation, -1, sizeof(INT) * entTotal);
    memset(rightIndexRelation2, -1, sizeof(INT) * relTotal);
    
    for (INT i = 1; i < trainTotal; i++) {
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
    rightIndexHead[trainingListHead[trainTotal - 1].h] = trainTotal - 1;
    leftIndexTail[trainingListTail[0].t] = 0;
    rightIndexTail[trainingListTail[trainTotal - 1].t] = trainTotal - 1;
    leftIndexRelation[trainingListRel[0].h] = 0;
    rightIndexRelation[trainingListRel[trainTotal - 1].h] = trainTotal - 1;
    leftIndexRelation2[trainingListRel2[0].r] = 0;
    rightIndexRelation2[trainingListRel2[trainTotal - 1].r] = trainTotal - 1;
    
    callocRealArray(leftIndex_mean, relTotal);
    callocRealArray(rightIndex_mean, relTotal);

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
void evolveTrainList() {
    resetIncrementalHelpers();

    INT operation = 0;
    if (numOperationsRate == 0)
        numOperationsRate = totalOperations;
    while(operation < numOperationsRate){
        if(KnowledgeGraphOperations[next_operation_id].operation == '+')
            insertTrainTriple(KnowledgeGraphOperations[next_operation_id].triple);
        
        else if(KnowledgeGraphOperations[next_operation_id].operation == '-')
            deleteTrainTriple(KnowledgeGraphOperations[next_operation_id].triple);
        
        operation++;
        next_operation_id++;

        if(next_operation_id == totalOperations){
            lastOperationFinished = true;
            printf("Reached snapshot.\n");
        }
    }
    
    loadIncrementalHelpers(
        trainList,
        trainHead,
        trainTail,
        trainRel,
        trainRel2,
        freqRel,
        freqEnt,
        trainTotal,
        entityTotal, // set to num_entities from global entity2id.txt ?
        relationTotal, // set to num_entities from global relation2id.txt ?
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

    printf("Currently contained train triples: %ld.\n", trainTotal);
}

extern "C"
void evolveTripleList() {
    INT operation = 0;
    if (numOperationsRate == 0)
        numOperationsRate = totalOperations;
        
    while(operation < numOperationsRate){
        if(KnowledgeGraphOperations[next_operation_id].operation == '+')
            insertTriple(KnowledgeGraphOperations[next_operation_id].triple);
        
        else if(KnowledgeGraphOperations[next_operation_id].operation == '-')
            deleteTriple(KnowledgeGraphOperations[next_operation_id].triple);
        
        operation++;
        next_operation_id++;
    }

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    printf("Currently contained entities: %ld.\n", getNumCurrentlyContainedEntities());
    printf("Currently deleted entities: %ld.\n", num_deleted_entities);
    printf("All entities: %ld.\n", num_all_entities);
    printf("Currently contained triples: %ld.\n", tripleTotal);
}


#endif

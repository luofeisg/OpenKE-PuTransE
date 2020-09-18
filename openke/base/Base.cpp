#include "Setting.h"
#include "Triple.h"
#include "Random.h"
#include "Reader.h"
#include "Incremental.h"
#include "Corrupt.h"
#include "Test.h"
#include "Valid.h"
#include "UniverseSetting.h"
#include "UniverseConstructor.h"
#include "Checks.h"
#include <cstdlib>
#include <set>
#include <pthread.h>

/*
===============Setting.h===============
*/

extern "C"
void print_con(REAL *con);

extern "C"
void test_balance_param(REAL balance, INT triple_constr);

extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

/*
===============Random.h===============
*/

extern "C"
void randReset();

extern "C"
void setRandomSeed(INT seed);

extern "C"
INT getRandomSeed();

/*
===============Reader.h===============
*/

extern "C"
void readGlobalNumEntities();

extern "C"
void readGlobalNumRelations();

extern "C"
void importTrainFiles();

extern "C"
void printTrainHead();

extern "C"
void getTrainingTriples(
        INT *headList,
        INT *relList,
        INT *tailList
);

extern "C"
void activateLoadOfAllTriples(bool);

/*
===============ParallelUniverse.h===============
*/

extern "C"
INT getEntityTotalUniverse();

extern "C"
INT getTrainTotalUniverse();

extern "C"
INT getRelationTotalUniverse();

extern "C"
void getParallelUniverse(
        INT triple_constraint,
        REAL balance_parameter);

extern "C"
void getEntityRemapping(INT *ent_remapping);

extern "C"
void getRelationRemapping(INT *rel_remapping);

extern "C"
void swapHelpers();

extern "C"
void resetUniverse();

extern "C"
void enableChecks();

/*
===============Incremental.h===============
*/
extern "C"
void setNumSnapshots(INT num);

extern "C"
INT getNumSnapshots();

extern "C"
void activateIncrementalSetting();

extern "C"
void initializeTrainingOperations(int snapshot);

extern "C"
void evolveTrainList();

extern "C"
void initializeTripleOperations(int snapshot);

extern "C"
void evolveTripleList();

extern "C"
INT getNumCurrentlyContainedEntities();

extern "C"
void loadTestData(int snapshot);

extern "C"
void loadValidData(int snapshot);

extern "C"
void loadSnapshotTriples(int snapshot);
/*
================================================
*/

struct Parameter {
    INT id;
    INT *batch_h;
    INT *batch_t;
    INT *batch_r;
    REAL *batch_y;
    INT batchSize;
    INT negRate;
    INT negRelRate;
    bool p;
    bool val_loss;
    INT mode;
    bool filter_flag;
};

void *getBatch(void *con) {
    Parameter *para = (Parameter *) (con);
    INT id = para->id;
    INT *batch_h = para->batch_h;
    INT *batch_t = para->batch_t;
    INT *batch_r = para->batch_r;
    REAL *batch_y = para->batch_y;
    INT batchSize = para->batchSize;
    INT negRate = para->negRate;
    INT negRelRate = para->negRelRate;
    bool p = para->p;
    bool val_loss = para->val_loss;
    INT mode = para->mode;
    bool filter_flag = para->filter_flag;
    INT lef, rig;
    if (batchSize % workThreads == 0) {
        lef = id * (batchSize / workThreads);
        rig = (id + 1) * (batchSize / workThreads);
    } else {
        lef = id * (batchSize / workThreads + 1);
        rig = (id + 1) * (batchSize / workThreads + 1);
        if (rig > batchSize) rig = batchSize;
    }
    REAL prob = 500;
    if (val_loss == false) {
        for (INT batch = lef; batch < rig; batch++) {
            INT i = rand_max(id, trainTotal);
            batch_h[batch] = trainList[i].h;
            batch_t[batch] = trainList[i].t;
            batch_r[batch] = trainList[i].r;
            batch_y[batch] = 1;
            INT last = batchSize;
            for (INT times = 0; times < negRate; times++) {
                if (mode == 0) {
                    if (bernFlag)
                        prob = 1000 * right_mean[trainList[i].r] /
                               (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
                    if (randd(id) % 1000 < prob) {
                        batch_h[batch + last] = trainList[i].h;
                        batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r, filter_flag);
                        batch_r[batch + last] = trainList[i].r;
                    } else {
                        batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r, filter_flag);
                        batch_t[batch + last] = trainList[i].t;
                        batch_r[batch + last] = trainList[i].r;
                    }
                    batch_y[batch + last] = -1;
                    last += batchSize;
                } else {
                    if (mode == -1) {
                        batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
                        batch_t[batch + last] = trainList[i].t;
                        batch_r[batch + last] = trainList[i].r;
                    } else {
                        batch_h[batch + last] = trainList[i].h;
                        batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
                        batch_r[batch + last] = trainList[i].r;
                    }
                    batch_y[batch + last] = -1;
                    last += batchSize;
                }
            }
            for (INT times = 0; times < negRelRate; times++) {
                batch_h[batch + last] = trainList[i].h;
                batch_t[batch + last] = trainList[i].t;
                batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
                batch_y[batch + last] = -1;
                last += batchSize;
            }
        }
    } else {
        for (INT batch = lef; batch < rig; batch++) {
            batch_h[batch] = validList[batch].h;
            batch_t[batch] = validList[batch].t;
            batch_r[batch] = validList[batch].r;
            batch_y[batch] = 1;
        }
    }
    pthread_exit(NULL);
}

extern "C"
void sampling(
        INT *batch_h,
        INT *batch_t,
        INT *batch_r,
        REAL *batch_y,
        INT batchSize,
        INT negRate = 1,
        INT negRelRate = 0,
        INT mode = 0,
        bool filter_flag = true,
        bool p = false,
        bool val_loss = false
) {
    pthread_t *pt = (pthread_t *) malloc(workThreads * sizeof(pthread_t));
    Parameter *para = (Parameter *) malloc(workThreads * sizeof(Parameter));
    for (INT threads = 0; threads < workThreads; threads++) {
        para[threads].id = threads;
        para[threads].batch_h = batch_h;
        para[threads].batch_t = batch_t;
        para[threads].batch_r = batch_r;
        para[threads].batch_y = batch_y;
        para[threads].batchSize = batchSize;
        para[threads].negRate = negRate;
        para[threads].negRelRate = negRelRate;
        para[threads].p = p;
        para[threads].val_loss = val_loss;
        para[threads].mode = mode;
        para[threads].filter_flag = filter_flag;
        pthread_create(&pt[threads], NULL, getBatch, (void *) (para + threads));
    }
    for (INT threads = 0; threads < workThreads; threads++)
        pthread_join(pt[threads], NULL);

    free(pt);
    free(para);
    
    if (checkOn){
        if(swap){
            checkSampling(batch_h, batch_t, batch_r, batch_y, trainListUniverse, trainTotal, batchSize);
        } else {
            checkSampling(batch_h, batch_t, batch_r, batch_y, trainList, trainTotal, batchSize);
        }
    }
}

extern "C"
INT getNumOfNegatives(INT entity, INT relation, bool entity_is_tail) {
    
    INT num_of_negative_entities = 0;
    if(entity_is_tail==1){
        
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r != relation)
                num_of_negative_entities += 1;
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r != relation)
                num_of_negative_entities += 1;
        }
    }
    return num_of_negative_entities;
}

extern "C"
INT getNumOfPositives(INT entity, INT relation, bool entity_is_tail) {
    INT num_of_positive_entities = 0;
    if(entity_is_tail){
        
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r == relation)
                num_of_positive_entities += 1;
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r == relation)
                num_of_positive_entities += 1;
        }
    }
    return num_of_positive_entities;
}

extern "C"
void getNegativeEntities(INT* batch_neg_entities, INT entity, INT relation, bool entity_is_tail) {
    INT batch_neg_entities_idx = 0;
    if(entity_is_tail){
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r != relation){
                batch_neg_entities[batch_neg_entities_idx] = trainTail[i].h; 
                batch_neg_entities_idx += 1;
            }    
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r != relation){
                batch_neg_entities[batch_neg_entities_idx] = trainHead[i].t; 
                batch_neg_entities_idx += 1;
            }
        }
    }
}

extern "C"
void getPositiveEntities(INT* batch_pos_ent, INT entity, INT relation, bool entity_is_tail) {
    INT batch_neg_entities_idx = 0;
    if(entity_is_tail){
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r == relation){
                batch_pos_ent[batch_neg_entities_idx] = trainTail[i].h; 
                batch_neg_entities_idx += 1;
            }    
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r == relation){
                batch_pos_ent[batch_neg_entities_idx] = trainHead[i].t; 
                batch_neg_entities_idx += 1;
            }
        }
    }
}

extern "C"
INT getNumOfEntityRelations(INT entity, bool entity_is_tail) {
    INT num_of_relations = 0;
    INT current_rel = -1;

    if(entity_is_tail==1){
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r != current_rel){
                current_rel = trainTail[i].r;
                num_of_relations += 1;
            }
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r != current_rel){
                current_rel = trainHead[i].r;
                num_of_relations += 1;
            }
        }
    }
    return num_of_relations;
}

extern "C"
void getEntityRelations(INT* entity_rels, INT entity, bool entity_is_tail) {
    INT entity_rels_idx = 0;
    INT current_rel = -1;

    if(entity_is_tail==1){
        INT lefIndexTail = lefTail[entity];
        INT rigIndexTail = rigTail[entity];
        
        for(int i=lefIndexTail; i <= rigIndexTail; i++){
            if(trainTail[i].r != current_rel){
                current_rel = trainTail[i].r;
                entity_rels[entity_rels_idx] = trainTail[i].r;
            }
        }
    }else{
        INT lefIndexHead = lefHead[entity];
        INT rigIndexHead = rigHead[entity];
        
        for(int i=lefIndexHead; i <= rigIndexHead; i++){
            if(trainHead[i].r != current_rel){
                current_rel = trainHead[i].r;
                entity_rels[entity_rels_idx] = trainHead[i].r;
            }
        }
    }
}

int main() {   
    
    // // Static
    // inPath = "../../benchmarks/Wikidata/WikidataEvolve/static/1/";
    // setBern(1);
    // setWorkThreads(8);
    // randReset();
    // setRandomSeed(4);
    // activateLoadOfAllTriples(true);
    // importTrainFiles();
    // importTestFiles();


    // INT batchSize = 5436;
    // INT batchSeqSize = batchSize * 2;

    // INT *batch_h; 
    // INT *batch_t;
    // INT *batch_r;
    // REAL *batch_y;
    
    

    // callocIntArray(batch_h,batchSeqSize);
    // callocIntArray(batch_t,batchSeqSize);
    // callocIntArray(batch_r,batchSeqSize);
    // callocRealArray(batch_y,batchSeqSize);
    
    // int epochs = 1000;
    // int batches = 100;
    // int iterations = 2 * (epochs + batches);
    // for(int i=0;i<iterations;i++){
    //     sampling(
    //     batch_h,
    //     batch_t,
    //     batch_r,
    //     batch_y,
    //     batchSize,
    //     1,
    //     0,
    //     0,
    //     true,
    //     false,
    //     false
    // ); 

    // }
    
    // Incremental
    // inPath = "../../benchmarks/Wikidata/WikidataEvolve/";
    // setBern(1);
    // setWorkThreads(8);
    // randReset();
    // setRandomSeed();


    //  setNumSnapshots(4);

    //  initializeIncrementalSetting();
    //  for(int i = 1; i<=num_snapshots; i++){
    //     // Evolve train list
    //     initializeTrainingOperations(i);
    //     evolveTrainList();
         
    //      // Evolve triple List
    //     loadSnapshotTriples(i);

    //     // Parallel Universe - gather embedding spaces
    //     INT triplet_constraint = trainTotal * 0.01;
    //     getParallelUniverse(triplet_constraint, 0.5);
    //     swapHelpers();
    //     resetUniverse();

    //  }
    return 0;
}
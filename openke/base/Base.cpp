#include "Setting.h"
#include "Triple.h"
#include "Random.h"
#include "Reader.h"
#include "ParallelUniverse.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <set>
#include <pthread.h>

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
INT getEntityTotalUniverse();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTrainTotalUniverse();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void initializeSingleRandomSeed();

extern "C"
void importTrainFiles();

extern "C"
void printTrainListUniverse();

extern "C"
void printTrainHead();

extern "C"
void getTrainUniverse(
        INT *head,
        INT *relation,
        INT *tail
);

extern "C"
void printTrainUniverse();

extern "C"
void getParallelUniverse(
        INT *headList,
        INT *relList,
        INT *tailList,
        INT *entity_remapping,
        INT *relation_remapping,
        INT triple_constraint,
        REAL balance_parameter,
        INT relation);

extern "C"
void swapHelpers();

extern "C"
void resetUniverse();

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
                        batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
                        batch_r[batch + last] = trainList[i].r;
                    } else {
                        batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
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
}


extern "C"
void getTrainingTriples(
        INT *headList,
        INT *relList,
        INT *tailList
) {
    for (int i = 0; i < trainTotal; i++) {
        headList[i] = trainList[i].h;
        relList[i] = trainList[i].r;
        tailList[i] = trainList[i].t;
    }
}

extern "C"
void getUniverseTrainingTriples(
        INT *headList,
        INT *relList,
        INT *tailList
) {
    for (int i = 0; i < trainTotalUniverse; i++) {
        headList[i] = trainListUniverse[i].h;
        relList[i] = trainListUniverse[i].r;
        tailList[i] = trainListUniverse[i].t;
    }
}

extern "C"
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
            std::cout << "Head: " << head << "\n";
            std::cout << "Tail: " << tail << "\n";
            std::cout << "Rel: " << rel << "\n";
            std::cout << "------" << rel << "\n";
        }
    }
    std::cout << "Not found triples: " << num_of_no_entry << ".";
}

int main() {
    importTrainFiles();
    return 0;
}

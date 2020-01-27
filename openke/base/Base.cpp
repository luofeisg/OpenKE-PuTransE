#include "Setting.h"
#include "Triple.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <set>
#include <pthread.h>

Triple *universeTrainList;

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

extern "C"
void randReset();

extern "C"
INT randBetween(INT a, INT b);

extern "C"
void importTrainFiles();

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

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
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
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
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
					if(mode == -1){
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
	}
	else
	{
		for (INT batch = lef; batch < rig; batch++)
		{
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
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
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
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);
	free(para);
}

//TODO handle duplicates
extern "C"
void getParallelUniverse(
    INT *headList,
	INT *relList,
	INT *tailList,
    INT triple_constraint,
    REAL balance_parameter
) {
    universeTrainList = (Triple *)calloc(triple_constraint, sizeof(Triple));
	//INT relation_focus = 160;
	INT relation_focus = randBetween(0,relationTotal); //sample relation r from R
    std::cout<<"relational focus is: "<<relation_focus<<std::endl;

    INT semantic_threshold = balance_parameter * triple_constraint; // calculate threshold for selection of relevant entities
    REAL prob = 500;

    //Gather relevant entities
    INT left_index = lefRel2[relation_focus];
    INT right_index = rigRel2[relation_focus];
    std::set<INT> entity_set;
    std::set<INT> tmp_set;
    std::set<INT>::iterator it;

    for(int idx=left_index;idx<right_index+1;idx++){
        entity_set.insert(trainRel2[idx].h);
        entity_set.insert(trainRel2[idx].t);
    }

    // pick random subset
    if(entity_set.size() > semantic_threshold){
       std::set<INT>::iterator random_entity;
       INT num_sub = 0;
       while(tmp_set.size() < semantic_threshold){
           random_entity = entity_set.begin();
           std::advance(random_entity, rand() % entity_set.size());
           tmp_set.insert(*random_entity);
           entity_set.erase(random_entity);
       }
       entity_set.swap(tmp_set);
       tmp_set.clear();
    }


    std::cout<<"relevant entities "<<entity_set.size()<<std::endl;

    INT num = 0;
    INT tmp_index = 0;
    INT entity = 0;
    bool duplicate = false;
    INT new_head = 0;
    INT new_rel = 0;
    INT new_tail = 0;


    while (num < triple_constraint){
         for(it = entity_set.begin(); it != entity_set.end() && num < triple_constraint;){
            entity = *it;
            if (rand() % 1000 < prob) {
                head:
                if(rigHead[entity]!= -1){
                    tmp_index = randBetween(lefHead[entity], rigHead[entity]+1);
                    new_head = trainHead[tmp_index].h;
                    new_rel = trainHead[tmp_index].r;
                    new_tail = trainHead[tmp_index].t;

                    tmp_set.insert(trainHead[tmp_index].t);
                } else {
                    goto tail;
                }
            } else {
                tail:
                if(rigTail[entity]!= -1){
                    tmp_index = randBetween(lefTail[entity], rigTail[entity]+1);
                    new_head = trainTail[tmp_index].h;
                    new_rel = trainTail[tmp_index].r;
                    new_tail = trainTail[tmp_index].t;

                    tmp_set.insert(trainHead[tmp_index].h);
                } else {
                    goto head;
                }
            }
            for(int i = 0; i < num; i++){
                if(new_head == headList[i] && new_rel == relList[i] && new_tail == tailList[i]){
                duplicate = true;
                }
            }if(duplicate){
               duplicate = false;
               it++;
               continue;
            }
            headList[num] = new_head;
            relList[num] = new_rel;
            tailList[num] = new_tail;
            universeTrainList[num].h = new_head;
            universeTrainList[num].r = new_rel;
            universeTrainList[num].t = new_tail;
            entity_set.erase(it++);
            num++;
        }
        entity_set.swap(tmp_set);
    }
}

int main() {
	importTrainFiles();
	return 0;
}
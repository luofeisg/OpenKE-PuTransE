#ifndef PARALLELUNIVERSE_H
#define PARALLELUNIVERSE_H

#include "Triple.h"
#include "Reader.h"
#include "Random.h"

INT trainTotalUniverse = 0;
INT entityTotalUniverse = 0;
INT relationTotalUniverse = 0;

bool swap = false;

INT *freqRelUniverse, *freqEntUniverse;
INT *lefHeadUniverse, *rigHeadUniverse;
INT *lefTailUniverse, *rigTailUniverse;
INT *lefRelUniverse, *rigRelUniverse;
INT *lefRel2Universe, *rigRel2Universe;
REAL *left_meanUniverse, *right_meanUniverse;

Triple *trainListUniverse;
Triple *trainHeadUniverse;
Triple *trainTailUniverse;
Triple *trainRelUniverse;
Triple *trainRel2Universe;

INT *entity_mapping;
INT *relation_mapping;
/*
INT *entity_remapping;
INT *relation_remapping;
*/
extern "C"
INT getEntityTotalUniverse() {
    return entityTotalUniverse;
}

extern "C"
INT getTrainTotalUniverse() {
    return trainTotalUniverse;
}



extern "C"
void swapHelpers() {
    INT *swap_freqRel = freqRel;
    freqRel = freqRelUniverse;
    freqRelUniverse = swap_freqRel;

    INT *swap_freqEnt = freqEnt;
    freqEnt = freqEntUniverse;
    freqEntUniverse = swap_freqEnt;

    INT *swap_lefHead = lefHead;
    lefHead = lefHeadUniverse;
    lefHeadUniverse = swap_lefHead;

    INT *swap_rigHead = rigHead;
    rigHead = rigHeadUniverse;
    rigHeadUniverse = swap_rigHead;

    INT *swap_lefTail = lefTail;
    lefTail = lefTailUniverse;
    lefTailUniverse = swap_lefTail;

    INT *swap_rigTail = rigTail;
    rigTail = rigTailUniverse;
    rigTailUniverse = swap_rigTail;

    INT *swap_lefRel = lefRel;
    lefRel = lefRelUniverse;
    lefRelUniverse = swap_lefRel;

    INT *swap_rigRel = rigRel;
    rigRel = rigRelUniverse;
    rigRelUniverse = swap_rigRel;

    INT *swap_lefRel2 = lefRel2;
    lefRel2 = lefRel2Universe;
    lefRel2Universe = swap_lefRel2;

    INT *swap_rigRel2 = rigRel2;
    rigRel2 = rigRel2Universe;
    rigRel2Universe = swap_rigRel2;

    REAL *swap_left_mean = left_mean;
    left_mean = left_meanUniverse;
    left_meanUniverse = swap_left_mean;

    REAL *swap_right_mean = right_mean;
    right_mean = right_meanUniverse;
    right_meanUniverse = swap_right_mean;

    Triple *swap_trainList = trainList;
    trainList = trainListUniverse;
    trainListUniverse = swap_trainList;

    Triple *swap_trainHead = trainHead;
    trainHead = trainHeadUniverse;
    trainHeadUniverse = swap_trainHead;

    Triple *swap_trainTail = trainTail;
    trainTail = trainTailUniverse;
    trainTailUniverse = swap_trainTail;

    Triple *swap_trainRel = trainRel;
    trainRel = trainRelUniverse;
    trainRelUniverse = swap_trainRel;

    Triple *swap_trainRel2 = trainRel2;
    trainRel2 = trainRel2Universe;
    trainRel2Universe = swap_trainRel2;

    INT swap_entityTotal = entityTotal;
    entityTotal = entityTotalUniverse;
    entityTotalUniverse = swap_entityTotal;

    INT swap_relationTotal = relationTotal;
    relationTotal = relationTotalUniverse;
    relationTotalUniverse = swap_relationTotal;

    INT swap_trainTotal = trainTotal;
    trainTotal = trainTotalUniverse;
    trainTotalUniverse = swap_trainTotal;

    if (swap)
        swap = false;
    else
        swap = true;
}


void checkAllocations() {

    if (trainTotalUniverse < 1)
        std::cout << "trainTotalUniverse is: " << trainTotalUniverse << std::endl;

    if (entityTotalUniverse < 1)
        std::cout << "entityTotalUniverse is: " << entityTotalUniverse << std::endl;

    if (relationTotalUniverse < 1)
        std::cout << "relationTotalUniverse is: " << relationTotalUniverse << std::endl;

    if (trainHeadUniverse == NULL) {
        std::cout << "trainHeadUniverse was not allocated" << std::endl;
        std::cout << "length is: " << trainTotalUniverse << std::endl;
    }
    if (trainTailUniverse == NULL) {
        std::cout << "trainTailUniverse was not allocated" << std::endl;
        std::cout << "length is: " << trainTotalUniverse << std::endl;
    }
    if (trainRelUniverse == NULL) {
        std::cout << "trainRelUniverse was not allocated" << std::endl;
        std::cout << "length is: " << trainTotalUniverse << std::endl;
    }
    if (trainRel2Universe == NULL) {
        std::cout << "trainRel2Universe was not allocated" << std::endl;
        std::cout << "length is: " << trainTotalUniverse << std::endl;
    }
    if (freqRelUniverse == NULL) {
        std::cout << "freqRelUniverse was not allocated" << std::endl;
        std::cout << "length is: " << relationTotalUniverse << std::endl;
    }
    if (freqEntUniverse == NULL) {
        std::cout << "freqEntUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotalUniverse << std::endl;
    }
    if (lefHeadUniverse == NULL) {
        std::cout << "lefHeadUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (rigHeadUniverse == NULL) {
        std::cout << "rigHeadUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (lefTailUniverse == NULL) {
        std::cout << "lefTailUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (rigTailUniverse == NULL) {
        std::cout << "rigTailUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (lefRelUniverse == NULL) {
        std::cout << "lefRelUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (rigRelUniverse == NULL) {
        std::cout << "rigRelUniverse was not allocated" << std::endl;
        std::cout << "length is: " << entityTotal << std::endl;
    }
    if (lefRel2Universe == NULL) {
        std::cout << "lefRel2Universe was not allocated" << std::endl;
        std::cout << "length is: " << relationTotal << std::endl;
    }
    if (rigRel2Universe == NULL) {
        std::cout << "rigRel2Universe was not allocated" << std::endl;
        std::cout << "length is: " << relationTotal << std::endl;
    }
}


extern "C"
void resetUniverse() {
    if (swap)
        swapHelpers();

    trainTotalUniverse = 0;
    entityTotalUniverse = 0;
    relationTotalUniverse = 0;
    if (trainListUniverse != NULL) {
        std::cout << "reset 1" << std::endl;
        free(trainListUniverse);
    }

    if (trainHeadUniverse != NULL) {
        std::cout << "reset 2" << std::endl;
        free(trainHeadUniverse);
    }

    if (trainTailUniverse != NULL) {
        std::cout << "reset 3" << std::endl;
        free(trainTailUniverse);
    }
    if (trainRelUniverse != NULL) {
        std::cout << "reset 4" << std::endl;
        free(trainRelUniverse);
    }
    if (trainRel2Universe != NULL) {
        std::cout << "reset 5" << std::endl;
        free(trainRel2Universe);
    }
    if (freqRelUniverse != NULL) {
        std::cout << "reset 6" << std::endl;
        free(freqRelUniverse);
    }
    if (freqEntUniverse != NULL) {
        std::cout << "reset 7" << std::endl;
        free(freqEntUniverse);
    }
    if (lefHeadUniverse != NULL) {
        std::cout << "reset 8" << std::endl;
        free(lefHeadUniverse);
    }
    if (rigHeadUniverse != NULL) {
        std::cout << "reset 9" << std::endl;
        free(rigHeadUniverse);
    }
    if (lefTailUniverse != NULL) {
        std::cout << "reset 10" << std::endl;
        free(lefTailUniverse);
    }
    if (rigTailUniverse != NULL) {
        std::cout << "reset 11" << std::endl;
        free(rigTailUniverse);
    }
    if (lefRelUniverse != NULL) {
        std::cout << "reset 12" << std::endl;
        free(lefRelUniverse);
    }
    if (rigRelUniverse != NULL) {
        std::cout << "reset 13" << std::endl;
        free(rigRelUniverse);
    }
    if (lefRel2Universe != NULL) {
        std::cout << "reset 14" << std::endl;
        free(lefRel2Universe);
    }
    if (rigRel2Universe != NULL) {
        std::cout << "reset 15" << std::endl;
        free(rigRel2Universe);
    }
    if (left_meanUniverse != NULL) {
        std::cout << "reset 16" << std::endl;
        free(left_meanUniverse);
    }
    if (right_meanUniverse != NULL) {
        std::cout << "reset 17" << std::endl;
        free(right_meanUniverse);
    }

    /*
    free(entity_mapping);
    free(relation_mapping);

    free(entity_remapping);
    free(relation_remapping);
    */
}

extern "C"
void printTrainUniverse() {
    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "trainListUniverse" << trainListUniverse[n].h << "," << trainListUniverse[n].r << ", "
                  << trainListUniverse[n].t << ". " << std::endl;

    }

    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "trainHeadUniverse" << trainHeadUniverse[n].h << "," << trainHeadUniverse[n].r << ", "
                  << trainHeadUniverse[n].t << ". " << std::endl;

    }

    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "trainTailUniverse" << trainTailUniverse[n].h << "," << trainTailUniverse[n].r << ", "
                  << trainTailUniverse[n].t << ". " << std::endl;
    }

    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "trainRelUniverse" << trainRelUniverse[n].h << "," << trainRelUniverse[n].r << ", "
                  << trainRelUniverse[n].t << ". " << std::endl;
    }

    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "trainRel2Universe" << trainRel2Universe[n].h << "," << trainRel2Universe[n].r << ", "
                  << trainRel2Universe[n].t << ". " << std::endl;

    }

    for (int n = 0; n < relationTotalUniverse; n++) {
        std::cout << "freqRelUniverse" << n << ":" << freqRelUniverse[n] << "." << std::endl;


    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "freqEntUniverse" << n << ":" << freqEntUniverse[n] << "." << std::endl;


    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "lefHeadUniverse" << n << ":" << lefHeadUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "rigHeadUniverse" << n << ":" << rigHeadUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "lefTailUniverse" << n << ":" << lefTailUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "rigTailUniverse" << n << ":" << rigTailUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "lefRelUniverse" << n << ":" << lefRelUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < entityTotalUniverse; n++) {
        std::cout << "rigRelUniverse" << n << ":" << rigRelUniverse[n] << "." << std::endl;

    }

    for (int n = 0; n < relationTotalUniverse; n++) {
        std::cout << "lefRel2Universe" << n << ":" << lefRel2Universe[n] << "." << std::endl;

    }


    for (int n = 0; n < relationTotalUniverse; n++) {
        std::cout << "rigRel2Universe" << n << ":" << rigRel2Universe[n] << "." << std::endl;
    }

    for (int n = 0; n < relationTotalUniverse; n++) {
        std::cout << "left_meanUniverse" << n << ":" << left_meanUniverse[n] << "." << std::endl;
    }

    for (int n = 0; n < relationTotalUniverse; n++) {
        std::cout << "right_meanUniverse" << n << ":" << right_meanUniverse[n] << "." << std::endl;
    }

    std::cout << "Train total" << ": " << trainTotalUniverse << "." << std::endl;
    std::cout << "Entity total" << ": " << entityTotalUniverse << "." << std::endl;
    std::cout << "Relation total" << ": " << relationTotalUniverse << "." << std::endl;
}


extern "C"
void printTrainListUniverse() {
    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "round1:" << n << std::endl;
        std::cout << "trainListUniverse" << trainListUniverse[n].h << "," << trainListUniverse[n].r << ", "
                  << trainListUniverse[n].t << ". " << std::endl;
    }
}

INT gatherTripleFromHead(INT entity, INT &new_head, INT &new_rel, INT &new_tail) {
    INT tmp_index = rand(lefHead[entity], rigHead[entity] + 1);
    new_head = trainHead[tmp_index].h;
    new_rel = trainHead[tmp_index].r;
    new_tail = trainHead[tmp_index].t;
    return new_tail;
}

INT gatherTripleFromTail(INT entity, INT &new_head, INT &new_rel, INT &new_tail) {
    INT tmp_index = rand(lefTail[entity], rigTail[entity] + 1);
    new_head = trainTail[tmp_index].h;
    new_rel = trainTail[tmp_index].r;
    new_tail = trainTail[tmp_index].t;
    return new_head;
}

std::set<INT> get_entity_subset(std::set<INT> entity_set, INT semantic_threshold) {
    std::set<INT> entity_subset;
    std::set<INT>::iterator random_entity;
    INT num_sub = 0;
    while (entity_subset.size() < semantic_threshold) {
        random_entity = entity_set.begin();
        std::advance(random_entity, rand() % entity_set.size());
        entity_subset.insert(*random_entity);
        entity_set.erase(random_entity);
    }

    return entity_subset;
}

std::set<INT> gatherRelationEntities(INT relation) {
    INT left_index = lefRel2[relation];
    INT right_index = rigRel2[relation];
    std::set<INT> entity_set;
    for (int idx = left_index; idx < right_index + 1; idx++) {
        entity_set.insert(trainRel2[idx].h);
        entity_set.insert(trainRel2[idx].t);
    }
    return entity_set;
}

bool checkDuplicate(INT head, INT rel, INT tail, INT last_index) {
    for (int i = 0; i < last_index; i++) {
        if (head == trainListUniverse[i].h && rel == trainListUniverse[i].r &&
            tail == trainListUniverse[i].t) {
            return true;
        }
    }
    return false;
}

void BidirectionalRandomWalk(std::set<INT> entity_set, INT *&headList, INT *&relList, INT *&tailList) {
    INT universe_index = 0;
    INT num_round = 0;
    INT current_entity = -1;

    INT last_duplicate_entity = -1;
    INT iter_duplicate_gathering_tolerance = 5;
    bool duplicate = false;

    std::set<INT> new_starting_points;
    std::set<INT> universe_entities;
    std::set<INT> universe_relations;
    universe_entities.insert(entity_set.begin(), entity_set.end());

    REAL prob = 500;
    while (universe_index < trainTotalUniverse) {
        /*it != entity_set.end() && universe_index < trainTotalUniverse; BUG? because when entitset end and universe_index is not 1000*/
        for (std::set<INT>::iterator it = entity_set.begin();
             it != entity_set.end() && universe_index < trainTotalUniverse;) {
            current_entity = *it;
            /*
            std::cout << "Start with Entity: " << entity << std::endl;
            std::cout << "lefh " << lefHead[entity] << std::endl;
            std::cout << "righ " << rigHead[entity] << std::endl;
            std::cout << "left " << lefTail[entity] << std::endl;
            std::cout << "rigt " << rigTail[entity] << std::endl;
            */
            INT new_head = 0;
            INT new_rel = 0;
            INT new_tail = 0;
            INT new_starting_entity = -1;
            // Determine if gather triple with entity as head or as tail
            if (rand() % 1000 < prob) {

                if (rigHead[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromHead(current_entity, new_head, new_rel, new_tail);
                } else if (rigTail[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromTail(current_entity, new_head, new_rel, new_tail);
                } else {
                    std::cout << "Error for Entity " << current_entity << std::endl;
                }
            } else {

                if (rigTail[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromTail(current_entity, new_head, new_rel, new_tail);
                } else if (rigHead[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromHead(current_entity, new_head, new_rel, new_tail);
                } else {
                    std::cout << "Error for Entity " << current_entity << std::endl;
                }

            }
/*
            std::cout << "head" << new_head << std::endl;
            std::cout << "tail" << new_tail << std::endl;
            std::cout << "rel" << new_rel << std::endl;
*/

            //check whether gathered triple was already collected
            if (checkDuplicate(new_head, new_rel, new_tail, universe_index)) {
                duplicate = false;
                if (last_duplicate_entity == current_entity) {
                    iter_duplicate_gathering_tolerance--;
                } else {
                    last_duplicate_entity = current_entity;
                }
                if (iter_duplicate_gathering_tolerance == 0) {
                    iter_duplicate_gathering_tolerance = 5;
                    it++;
                }
/*
                std::cout << "relational focus: " << relation_focus << std::endl;
                std::cout << "entity: " << entity << std::endl;
                std::cout << "iteration step: " << universe_index << std::endl;
                std::cout << "iteration epoch: " << num_round << std::endl;
                std::cout << "Triple (h,r,t): (" << new_head << ", " << new_rel << ", " << new_tail << ")."
                          << std::endl;
                std::cout << "________________________________" << std::endl;
 */             continue;
            }

            headList[universe_index] = new_head;
            relList[universe_index] = new_rel;
            tailList[universe_index] = new_tail;
            trainListUniverse[universe_index].h = new_head;
            trainListUniverse[universe_index].r = new_rel;
            trainListUniverse[universe_index].t = new_tail;
            new_starting_points.insert(new_starting_entity);
            universe_entities.insert(new_starting_entity);
            universe_relations.insert(new_rel);
            entity_set.erase(it++);
            universe_index++;

        }
        entity_set.swap(new_starting_points);
        num_round++;
    }
    // set number of entities and relations in universe
    entityTotalUniverse = universe_entities.size();
    relationTotalUniverse = universe_relations.size();
}

void enumerateTrainUniverseTriples(INT *&entity_remapping, INT *&relation_remapping) {
    entity_mapping = (INT *) calloc(entityTotal, sizeof(INT));
    relation_mapping = (INT *) calloc(relationTotal, sizeof(INT));
    memset(entity_mapping, -1, sizeof(INT) * entityTotal);
    memset(relation_mapping, -1, sizeof(INT) * relationTotal);

    /*
    entity_remapping = (INT *) calloc(entityTotal, sizeof(INT));
    relation_remapping = (INT *) calloc(relationTotal, sizeof(INT));
    */
    memset(entity_remapping, -1, sizeof(INT) * entityTotal);
    memset(relation_remapping, -1, sizeof(INT) * relationTotal);

    INT next_entity_id = 0;
    INT next_relation_id = 0;

    for (INT i = 0; i < trainTotalUniverse; i++) {
        if (entity_mapping[trainListUniverse[i].h] == -1) {
            entity_mapping[trainListUniverse[i].h] = next_entity_id;
            entity_remapping[next_entity_id] = trainListUniverse[i].h;
            next_entity_id++;
        }
        trainListUniverse[i].h = entity_mapping[trainListUniverse[i].h];


        if (entity_mapping[trainListUniverse[i].t] == -1) {
            entity_mapping[trainListUniverse[i].t] = next_entity_id;
            entity_remapping[next_entity_id] = trainListUniverse[i].t;
            next_entity_id++;
        }
        trainListUniverse[i].t = entity_mapping[trainListUniverse[i].t];


        if (relation_mapping[trainListUniverse[i].r] == -1) {
            relation_mapping[trainListUniverse[i].r] = next_relation_id;
            relation_remapping[next_relation_id] = trainListUniverse[i].r;
            next_relation_id++;
        }
        trainListUniverse[i].r = relation_mapping[trainListUniverse[i].r];
    }
    free(entity_mapping);
    free(relation_mapping);
}

extern "C"
void getParallelUniverse(
        INT *headList,
        INT *relList,
        INT *tailList,
        INT *entity_remapping,
        INT *relation_remapping,
        INT triple_constraint,
        REAL balance_parameter,
        INT relation_focus
) {
    trainTotalUniverse = triple_constraint;
    trainListUniverse = (Triple *) calloc(trainTotalUniverse, sizeof(Triple));
    if (trainListUniverse == NULL) {
        std::cout << "trainListUniverse was not allocated" << std::endl;
        std::cout << "trainTotalUniverse: " << trainTotalUniverse << std::endl;
    }
    //INT relation_focus = 160;
    //INT relation_focus = 74;
    //INT relation_focus = rand(0, relationTotal); //sample relation r from R

    std::cout << "semantic focus is relation: " << relation_focus << "." << std::endl;
    INT semantic_threshold =
            balance_parameter * trainTotalUniverse; // calculate threshold for selection of relevant entities


    //Gather entities for semantic focus
    std::set<INT> entity_set = gatherRelationEntities(relation_focus);



    // pick random subset if len(entity_set) > semantic_threshold
    if (entity_set.size() > semantic_threshold) {
        entity_set = get_entity_subset(entity_set, semantic_threshold);
    }
    std::cout << "Entities gathered for semantic focus: " << entity_set.size() << std::endl;

    std::cout << "Gather training dataset for Universe..." << std::endl;
    BidirectionalRandomWalk(entity_set, headList, relList, tailList);
    std::cout << "Gathering of train entities finished." << std::endl;


    std::cout << "Enumerate entities and relations in triples." << std::endl;
    enumerateTrainUniverseTriples(entity_remapping, relation_remapping);

    printTrainListUniverse();

    std::cout << "Initialize helper arrays." << std::endl;
    initializeHelpers(
            trainListUniverse,
            trainHeadUniverse,
            trainTailUniverse,
            trainRelUniverse,
            trainRel2Universe,
            freqRelUniverse,
            freqEntUniverse,
            trainTotalUniverse,
            lefHeadUniverse,
            rigHeadUniverse,
            lefTailUniverse,
            rigTailUniverse,
            lefRelUniverse,
            rigRelUniverse,
            lefRel2Universe,
            rigRel2Universe,
            left_meanUniverse,
            right_meanUniverse
    );
    std::cout << "Universe configured." << std::endl;
    std::cout << "___________________________" << std::endl;
    printTrainUniverse();
    //resetUniverse();
    //return test_variable

}


extern "C"
void getTrainUniverse(
        INT *head,
        INT *relation,
        INT *tail
) {
    for (int n = 0; n < trainTotalUniverse; n++) {
        std::cout << "round " << n << " ." << std::endl;
        head[n] = trainListUniverse[n].h;
        relation[n] = trainListUniverse[n].r;
        tail[n] = trainListUniverse[n].t;
    }
}


#endif

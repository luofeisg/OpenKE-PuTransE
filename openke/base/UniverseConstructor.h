#ifndef UNIVERSEMETHODS_H
#define UNIVERSEMETHODS_H

#include "Triple.h"
#include "Reader.h"
#include "Random.h"
#include "UniverseSetting.h"
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>

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

void BidirectionalRandomWalk(std::set<INT> entity_set) {
    INT universe_index = 0;
    INT num_round = 0;
    INT current_entity = -1;

    INT last_duplicate_entity = -1;
    INT iter_duplicate_gathering_tolerance = 5;
    bool duplicate = false;

    std::set<INT> new_starting_points;
    std::set<INT> universe_entities;
    std::set<INT> universe_relations;

    REAL prob = 500;
    while (universe_index < trainTotalUniverse) {
        for (std::set<INT>::iterator it = entity_set.begin();
             it != entity_set.end() && universe_index < trainTotalUniverse;) {
            current_entity = *it;

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
                    printf("Error for Entity: %ld.\n", current_entity);
                }
            } else {
                if (rigTail[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromTail(current_entity, new_head, new_rel, new_tail);
                } else if (rigHead[current_entity] != -1) {
                    new_starting_entity = gatherTripleFromHead(current_entity, new_head, new_rel, new_tail);
                } else {
                    printf("Error for Entity: %ld.\n", current_entity);
                }
            }

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
                continue;
            }

            trainListUniverse[universe_index].h = new_head;
            trainListUniverse[universe_index].r = new_rel;
            trainListUniverse[universe_index].t = new_tail;

            new_starting_points.insert(new_starting_entity);
            universe_entities.insert(new_tail);
            universe_entities.insert(new_head);
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

void enumerateTrainUniverseTriples() {
    callocTripleArray(trainListUniverseEnum, trainTotalUniverse);

    INT *entity_mapping = (INT *) calloc(entityTotal, sizeof(INT));
    INT *relation_mapping = (INT *) calloc(relationTotal, sizeof(INT));
    memset(entity_mapping, -1, sizeof(INT) * entityTotal);
    memset(relation_mapping, -1, sizeof(INT) * relationTotal);

    callocIntArray(entity_remapping, entityTotal);
    callocIntArray(relation_remapping, relationTotal);
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
        trainListUniverseEnum[i].h = entity_mapping[trainListUniverse[i].h];


        if (entity_mapping[trainListUniverse[i].t] == -1) {
            entity_mapping[trainListUniverse[i].t] = next_entity_id;
            entity_remapping[next_entity_id] = trainListUniverse[i].t;
            next_entity_id++;
        }
        trainListUniverseEnum[i].t = entity_mapping[trainListUniverse[i].t];


        if (relation_mapping[trainListUniverse[i].r] == -1) {
            relation_mapping[trainListUniverse[i].r] = next_relation_id;
            relation_remapping[next_relation_id] = trainListUniverse[i].r;
            next_relation_id++;
        }
        trainListUniverseEnum[i].r = relation_mapping[trainListUniverse[i].r];
    }
}

void loadUniverseHelpers() {
    std::sort(trainListUniverseEnum, trainListUniverseEnum + trainTotalUniverse, Triple::cmp_head);

    callocTripleArray(trainHeadUniverse, trainTotal);
    callocTripleArray(trainTailUniverse, trainTotal);
    callocTripleArray(trainRelUniverse, trainTotal);
    callocTripleArray(trainRel2Universe, trainTotal);

    callocIntArray(freqEntUniverse, entityTotal);
    callocIntArray(freqRelUniverse, relationTotal);

    trainHeadUniverse[0] = trainTailUniverse[0] = trainRelUniverse[0] = trainRel2Universe[0] = trainListUniverseEnum[0];
    freqEntUniverse[trainListUniverseEnum[0].t] += 1;
    freqEntUniverse[trainListUniverseEnum[0].h] += 1;
    freqRelUniverse[trainListUniverseEnum[0].r] += 1;

    for (INT i = 1; i < trainTotalUniverse; i++) {
        trainHeadUniverse[i] = trainTailUniverse[i] = trainRelUniverse[i] = trainRel2Universe[i] = trainListUniverseEnum[i];
        freqEntUniverse[trainListUniverseEnum[i].t]++;
        freqEntUniverse[trainListUniverseEnum[i].h]++;
        freqRelUniverse[trainListUniverseEnum[i].r]++;
    }
    std::sort(trainHeadUniverse, trainHeadUniverse + trainTotalUniverse, Triple::cmp_head);
    std::sort(trainTailUniverse, trainTailUniverse + trainTotalUniverse, Triple::cmp_tail);
    std::sort(trainRelUniverse, trainRelUniverse + trainTotalUniverse, Triple::cmp_rel);
    std::sort(trainRel2Universe, trainRel2Universe + trainTotalUniverse, Triple::cmp_rel2);

    callocIntArray(lefHeadUniverse, entityTotal);
    callocIntArray(rigHeadUniverse, entityTotal);
    callocIntArray(lefTailUniverse, entityTotal);
    callocIntArray(rigTailUniverse, entityTotal);
    callocIntArray(lefRelUniverse, entityTotal);
    callocIntArray(rigRelUniverse, entityTotal);
    callocIntArray(lefRel2Universe, relationTotal);
    callocIntArray(rigRel2Universe, relationTotal);

    memset(rigHeadUniverse, -1, sizeof(INT) * entityTotal);
    memset(rigTailUniverse, -1, sizeof(INT) * entityTotal);
    memset(rigRelUniverse, -1, sizeof(INT) * entityTotal);
    memset(rigRel2Universe, -1, sizeof(INT) * relationTotal);

    for (INT i = 1; i < trainTotalUniverse; i++) {
        if (trainTailUniverse[i].t != trainTailUniverse[i - 1].t) {
            rigTailUniverse[trainTailUniverse[i - 1].t] = i - 1;
            lefTailUniverse[trainTailUniverse[i].t] = i;
        }
        if (trainHeadUniverse[i].h != trainHeadUniverse[i - 1].h) {
            rigHeadUniverse[trainHeadUniverse[i - 1].h] = i - 1;
            lefHeadUniverse[trainHeadUniverse[i].h] = i;
        }
        if (trainRelUniverse[i].h != trainRelUniverse[i - 1].h) {
            rigRelUniverse[trainRelUniverse[i - 1].h] = i - 1;
            lefRelUniverse[trainRelUniverse[i].h] = i;
        }
        if (trainRel2Universe[i].r != trainRel2Universe[i - 1].r) {
            rigRel2Universe[trainRel2Universe[i - 1].r] = i - 1;
            lefRel2Universe[trainRel2Universe[i].r] = i;
        }
    }

    lefHeadUniverse[trainHeadUniverse[0].h] = 0;
    rigHeadUniverse[trainHeadUniverse[trainTotalUniverse - 1].h] = trainTotalUniverse - 1;
    lefTailUniverse[trainTailUniverse[0].t] = 0;
    rigTailUniverse[trainTailUniverse[trainTotalUniverse - 1].t] = trainTotalUniverse - 1;
    lefRelUniverse[trainRelUniverse[0].h] = 0;
    rigRelUniverse[trainRelUniverse[trainTotalUniverse - 1].h] = trainTotalUniverse - 1;
    lefRel2Universe[trainRel2Universe[0].r] = 0;
    rigRel2Universe[trainRel2Universe[trainTotalUniverse - 1].r] = trainTotalUniverse - 1;

    callocRealArray(left_meanUniverse, relationTotal);
    callocRealArray(right_meanUniverse, relationTotal);

    for (INT i = 0; i < entityTotalUniverse; i++) {
        for (INT j = lefHeadUniverse[i] + 1; j <= rigHeadUniverse[i]; j++) {
            if (trainHeadUniverse[j].r != trainHeadUniverse[j - 1].r)
                left_meanUniverse[trainHeadUniverse[j].r] += 1.0;
        }
        if (lefHeadUniverse[i] <= rigHeadUniverse[i])
            left_meanUniverse[trainHeadUniverse[lefHeadUniverse[i]].r] += 1.0;
        for (INT j = lefTailUniverse[i] + 1; j <= rigTailUniverse[i]; j++) {
            if (trainTailUniverse[j].r != trainTailUniverse[j - 1].r)
                right_meanUniverse[trainTailUniverse[j].r] += 1.0;
        }
        if (lefTailUniverse[i] <= rigTailUniverse[i])
            right_meanUniverse[trainTailUniverse[lefTailUniverse[i]].r] += 1.0;
    }
    for (INT i = 0; i < relationTotalUniverse; i++) {
        left_meanUniverse[i] = freqRelUniverse[i] / left_meanUniverse[i];
        right_meanUniverse[i] = freqRelUniverse[i] / right_meanUniverse[i];
    }
}

extern "C"
void getParallelUniverse(
        INT triple_constraint,
        REAL balance_parameter,
        INT relation_focus
) {
    trainTotalUniverse = triple_constraint;
    callocTripleArray(trainListUniverse, trainTotalUniverse);

    //INT relation_focus = 160;
    //INT relation_focus = 74;
    //INT relation_focus = rand(0, relationTotal); //sample relation r from R

    printf("Semantic focus is relation: %ld.\n", relation_focus);
    INT semantic_threshold =
            balance_parameter * triple_constraint; // calculate threshold for selection of relevant entities

    //Gather entities for semantic focus
    std::set<INT> entity_set = gatherRelationEntities(relation_focus);

    // pick random subset if len(entity_set) > semantic_threshold
    if (entity_set.size() > semantic_threshold) {
        entity_set = get_entity_subset(entity_set, semantic_threshold);
    }
    printf("Entities gathered which participate in sampled focus: %ld.\n", entity_set.size());

    printf("Gather training dataset for Universe...\n");
    BidirectionalRandomWalk(entity_set);
    entity_set.clear();
    checkUniverseTrainingTriples();

    printf("Enumerate entities and relations in triples. \n");
    enumerateTrainUniverseTriples();
    checkEnumeration();

    printf("Initialize helper arrays. \n");
    loadUniverseHelpers();
    // loadHelpers(
    //         trainListUniverseEnum,
    //         trainHeadUniverse,
    //         trainTailUniverse,
    //         trainRelUniverse,
    //         trainRel2Universe,
    //         freqRelUniverse,
    //         freqEntUniverse,
    //         trainTotalUniverse,
    //         entityTotalUniverse,
    //         relationTotalUniverse,
    //         lefHeadUniverse,
    //         rigHeadUniverse,
    //         lefTailUniverse,
    //         rigTailUniverse,
    //         lefRelUniverse,
    //         rigRelUniverse,
    //         lefRel2Universe,
    //         rigRel2Universe,
    //         left_meanUniverse,
    //         right_meanUniverse
    // );
    checkUniverseHelpers();
    printf("Universe configured. \n");
}

#endif
#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Incremental.h"

/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
INT lastRel = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

REAL l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
REAL l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
REAL hit1, hit3, hit10, mr, mrr;
REAL hit1TC, hit3TC, hit10TC, mrTC, mrrTC;

extern "C"
void initTest() {
    printf("Initialize test");
    lastHead = 0;
    lastTail = 0;
    lastRel = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
    REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

    l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
    l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
}

// TODO test
extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
    // Attach test head entity to beginning of batch array
    INT testH = testList[lastHead].h; 
    INT testT = testList[lastHead].t;
    INT testR = testList[lastHead].r;
    ph[0] = testH;
    pt[0] = testT;
    pr[0] = testR;
    INT offset = -1;

    if(incrementalSetting){
        // INT i = 1;
        // for(INT entity : currently_contained_entities) {
        //     // TODO Test
        //     if (entity == testH)
        //         continue;
        
        // TODO: Add not only currently contained entities but also entities from test/ valid triples?
        // But all entities from test/ valid have to be hold in the training data to be represented during the
        // Training Process
        //
        // What if triple is deleted in training data whereby also an entity is deleted, and at the same time 
        // the entity is existing in a triple in test/ valid data?
        // How to handle that:
        // 1) Including scanning of test/ valid data in the checking of an deleted entity
        //    after a triple deletion takes places. Implies that we load test/ valid data
        //    before we load a snapshot, so together with the training data (and not seperately and after that).
        //    This would also mean, that an entity could be completely deleted in the train data, although it
        //    could be included in the test data. Further it would occur in the test/valid (head-/tail-) batch 
        //    Would that be sufficient?
        //
        // 2) Make sure that after compilation train-/valid-/test data is consistent. So that we reload the
        //    'currently_contained_entities' array by importing these datasets from the static folder.
        //    


        for(INT i=1; i<num_currently_contained_entities;i++) {
            if (currently_contained_entities[i+offset] == testH)
                offset++;
            
            INT entity = currently_contained_entities[i + offset];

            ph[i] = entity;
            pt[i] = testT;
            pr[i] = testR;
            i++;
        } 
    }else{
        for (INT i = 1; i < entityTotal; i++) {
            if (i + offset == testH)
                offset++;

            ph[i] = i + offset;
            pt[i] = testT;
            pr[i] = testR;
        }
    }
    lastHead++;
}

extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr) {
    INT testH = testList[lastTail].h; 
    INT testT = testList[lastTail].t;
    INT testR = testList[lastTail].r;
    INT offset = -1;

    ph[0] = testH;
    pt[0] = testT;
    pr[0] = testR;
    if(incrementalSetting){
        for(INT i=1; i<num_currently_contained_entities;i++) {
            // TODO Test
            if(currently_contained_entities[i + offset] == testT)
                offset++;
            
            INT entity = currently_contained_entities[i + offset]; 

            ph[i] = testH;
            pt[i] = entity;
            pr[i] = testR;
            i++;
        } 
    }else{
        for (INT i = 1; i < entityTotal; i++) {
            if (i + offset == testT)
                offset++;
            
            ph[i] = testH;
            pt[i] = i + offset;
            pr[i] = testR;
        }
    }
    lastTail++;
}

extern "C"
void getRelBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < relationTotal; i++) {
        ph[i] = testList[lastRel].h;
        pt[i] = testList[lastRel].t;
        pr[i] = i;
    }
}

extern "C"
void testHead(REAL *con, INT lastHead, bool type_constrain = false) {
    printf("lastHead: %ld.\n", lastHead);
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;
    INT filter_offset = -1;

    INT lef, rig;
    if (type_constrain) {
        lef = head_lef[r];
        rig = head_rig[r];
    }
    REAL minimal = con[0];
    printf("lastHead score: %f.\n", con[0]);
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    if (minimal != INFINITY){
        if (incrementalSetting){
            for (INT j = 1; j < num_currently_contained_entities; j++) {
                REAL value = con[j];
                if (currently_contained_entities[j + filter_offset] == h)
                        filter_offset++;
                    
                if (value < minimal) {
                    l_s += 1;
                    
                    if (not _find(currently_contained_entities[j + filter_offset], t, r)){
                        l_filter_s += 1;
                    }
                }
            }
        } else {
            for (INT j = 1; j < entityTotal; j++) {
                REAL value = con[j];
                if (j + filter_offset == h)
                        filter_offset++;
                    
                if (value < minimal) {
                    l_s += 1;
                    
                    if (not _find(j + filter_offset, t, r)){
                        l_filter_s += 1;
                    }else{
                        printf("Found: %ld, %ld, %ld.\n", j + filter_offset, t, r);
                    }
                }
                if (type_constrain) {
                    while (lef < rig && head_type[lef] < j) lef ++;
                    if (lef < rig && j == head_type[lef]) {
                        if (value < minimal) {
                            l_s_constrain += 1;
                            if (not _find(j, t, r)) {
                                l_filter_s_constrain += 1;
                            }
                        }  
                    }
                }
            }
        }
    } else {
        if (incrementalSetting){
            l_s = num_currently_contained_entities;
            l_filter_s = num_currently_contained_entities;

            for (INT j = 1; j < num_currently_contained_entities; j++) {
                if (currently_contained_entities[j + filter_offset] == h)
                    filter_offset++;
                                
                if (not _find(currently_contained_entities[j + filter_offset], t, r)){
                    l_filter_s -= 1;
                }
            }
        } else {
            l_s = entityTotal;
            l_filter_s = entityTotal;
            
            for (INT j = 1; j < entityTotal; j++){ 
                if (j + filter_offset == h)
                        filter_offset++;
                
                if (not _find(j + filter_offset, t, r))
                    l_filter_s -= 1;
                
            }
        }
    } 

    printf("Triple (%ld, %ld, %ld).\n", h, t, r);
    printf("raw Rank: %ld.\n", l_s);
    printf("filter Rank: %ld.\n", l_filter_s);
    printf("-------\n");
    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1 + l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);

    if (type_constrain) {
        if (l_filter_s_constrain < 10) l_filter_tot_constrain += 1;
        if (l_s_constrain < 10) l_tot_constrain += 1;
        if (l_filter_s_constrain < 3) l3_filter_tot_constrain += 1;
        if (l_s_constrain < 3) l3_tot_constrain += 1;
        if (l_filter_s_constrain < 1) l1_filter_tot_constrain += 1;
        if (l_s_constrain < 1) l1_tot_constrain += 1;

        l_filter_rank_constrain += (l_filter_s_constrain+1);
        l_rank_constrain += (1+l_s_constrain);
        l_filter_reci_rank_constrain += 1.0/(l_filter_s_constrain+1);
        l_reci_rank_constrain += 1.0/(l_s_constrain+1);
    }
}

extern "C"
void testTail(REAL *con, INT lastTail, bool type_constrain = false) {
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    INT filter_offset = -1;

    INT lef, rig;
    if (type_constrain) {
        lef = tail_lef[r];
        rig = tail_rig[r];
    }
    REAL minimal = con[0];
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;
    
    if (minimal != INFINITY){
        if (incrementalSetting){
            for (INT j = 1; j < num_currently_contained_entities; j++) {
                REAL value = con[j];
                if (currently_contained_entities[j + filter_offset] == t)
                        filter_offset++;
                    
                if (value < minimal) {
                    r_s += 1;
                    
                    if (not _find(h, currently_contained_entities[j + filter_offset], r)){
                        r_filter_s += 1;
                    }
                }
            }
        } else {
            for (INT j = 1; j < entityTotal; j++) {
                REAL value = con[j];
                if (j + filter_offset == t)
                        filter_offset++;    
                    
                if (value < minimal) {
                    r_s += 1;                

                    if (not _find(h, j + filter_offset, r)){
                        r_filter_s += 1;
                    } else {
                        printf("Found: %ld, %ld, %ld.\n", h, j + filter_offset, r);
                    }
                }
                if (type_constrain) {
                    while (lef < rig && tail_type[lef] < j) lef ++;
                    if (lef < rig && j == tail_type[lef]) {
                            if (value < minimal) {
                                r_s_constrain += 1;
                                if (not _find(h, j ,r)) {
                                    r_filter_s_constrain += 1;
                                }
                            }
                    }
                }
                
            }
        }
    } else {
        if (incrementalSetting){
            r_s = num_currently_contained_entities;
            r_filter_s = num_currently_contained_entities;    

            for (INT j = 1; j < num_currently_contained_entities; j++) {
                if (currently_contained_entities[j + filter_offset] == t)
                    filter_offset++;
                
                if (not _find(h, currently_contained_entities[j + filter_offset], r))
                    r_filter_s -= 1;
            }
        } else {
            r_s = entityTotal;
            r_filter_s = entityTotal;
            for (INT j = 1; j < entityTotal; j++){ 
                if (j + filter_offset == t)
                        filter_offset++;
                
                if (not _find(h, j + filter_offset, r))
                    r_filter_s -= 1;
            }
        }
    }
    printf("Triple (%ld, %ld, %ld).\n", h, t, r);
    printf("raw Rank: %ld.\n", r_s);
    printf("filter Rank: %ld.\n", r_filter_s);
    printf("-------\n");
    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);
    
    if (type_constrain) {
        if (r_filter_s_constrain < 10) r_filter_tot_constrain += 1;
        if (r_s_constrain < 10) r_tot_constrain += 1;
        if (r_filter_s_constrain < 3) r3_filter_tot_constrain += 1;
        if (r_s_constrain < 3) r3_tot_constrain += 1;
        if (r_filter_s_constrain < 1) r1_filter_tot_constrain += 1;
        if (r_s_constrain < 1) r1_tot_constrain += 1;

        r_filter_rank_constrain += (1+r_filter_s_constrain);
        r_rank_constrain += (1+r_s_constrain);
        r_filter_reci_rank_constrain += 1.0/(1+r_filter_s_constrain);
        r_reci_rank_constrain += 1.0/(1+r_s_constrain);
    }
}

extern "C"
void testRel(REAL *con) {
    INT h = testList[lastRel].h;
    INT t = testList[lastRel].t;
    INT r = testList[lastRel].r;

    REAL minimal = con[r];
    INT rel_s = 0;
    INT rel_filter_s = 0;

    for (INT j = 0; j < relationTotal; j++) {
        if (j != r) {
            REAL value = con[j];
            if (value < minimal) {
                rel_s += 1;
                if (not _find(h, t, j))
                    rel_filter_s += 1;
            }
        }
    }

    if (rel_filter_s < 10) rel_filter_tot += 1;
    if (rel_s < 10) rel_tot += 1;
    if (rel_filter_s < 3) rel3_filter_tot += 1;
    if (rel_s < 3) rel3_tot += 1;
    if (rel_filter_s < 1) rel1_filter_tot += 1;
    if (rel_s < 1) rel1_tot += 1;

    rel_filter_rank += (rel_filter_s+1);
    rel_rank += (1+rel_s);
    rel_filter_reci_rank += 1.0/(rel_filter_s+1);
    rel_reci_rank += 1.0/(rel_s+1);

    lastRel++;
}


extern "C"
void test_link_prediction(bool type_constrain = false) {
    printf("Test Total is: %ld.\n", testTotal);
    printf("Triple Total is: %ld.\n", tripleTotal);
    printf("l_rank is: %f.\n", l_rank);
    printf("r_rank is: %f.\n", r_rank);
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;
 
    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;
 
    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    // with filter
    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;
 
    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;
 
    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit10 = (l_filter_tot+r_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;

    if (type_constrain) {
        //type constrain
        l_rank_constrain /= testTotal;
        r_rank_constrain /= testTotal;
        l_reci_rank_constrain /= testTotal;
        r_reci_rank_constrain /= testTotal;
     
        l_tot_constrain /= testTotal;
        l3_tot_constrain /= testTotal;
        l1_tot_constrain /= testTotal;
     
        r_tot_constrain /= testTotal;
        r3_tot_constrain /= testTotal;
        r1_tot_constrain /= testTotal;

        // with filter
        l_filter_rank_constrain /= testTotal;
        r_filter_rank_constrain /= testTotal;
        l_filter_reci_rank_constrain /= testTotal;
        r_filter_reci_rank_constrain /= testTotal;
     
        l_filter_tot_constrain /= testTotal;
        l3_filter_tot_constrain /= testTotal;
        l1_filter_tot_constrain /= testTotal;
     
        r_filter_tot_constrain /= testTotal;
        r3_filter_tot_constrain /= testTotal;
        r1_filter_tot_constrain /= testTotal;

        printf("type constraint results:\n");
        
        printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
        printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain);
        printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain);
        printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
                (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2);
        printf("\n");
        printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain);
        printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain);
        printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
                (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2);

        mrrTC = (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2;
        mrTC = (l_filter_rank_constrain+r_filter_rank_constrain) / 2;
        hit10TC = (l_filter_tot_constrain+r_filter_tot_constrain) / 2;
        hit3TC = (l3_filter_tot_constrain+r3_filter_tot_constrain) / 2;
        hit1TC = (l1_filter_tot_constrain+r1_filter_tot_constrain) / 2;
    }
}

extern "C"
void test_relation_prediction() {
    rel_rank /= testTotal;
    rel_reci_rank /= testTotal;
  
    rel_tot /= testTotal;
    rel3_tot /= testTotal;
    rel1_tot /= testTotal;

    // with filter
    rel_filter_rank /= testTotal;
    rel_filter_reci_rank /= testTotal;
  
    rel_filter_tot /= testTotal;
    rel3_filter_tot /= testTotal;
    rel1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            rel_reci_rank, rel_rank, rel_tot, rel3_tot, rel1_tot);
    printf("\n");
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            rel_filter_reci_rank, rel_filter_rank, rel_filter_tot, rel3_filter_tot, rel1_filter_tot);
}

extern "C"
REAL getTestLinkHit10(bool type_constrain = false) {
    if (type_constrain)
        return hit10TC;
    printf("%f\n", hit10);
    return hit10;
}

extern "C"
REAL  getTestLinkHit3(bool type_constrain = false) {
    if (type_constrain)
        return hit3TC;
    return hit3;
}

extern "C"
REAL  getTestLinkHit1(bool type_constrain = false) {
    if (type_constrain)
        return hit1TC;    
    return hit1;
}

extern "C"
REAL  getTestLinkMR(bool type_constrain = false) {
    if (type_constrain)
        return mrTC;
    return mr;
}

extern "C"
REAL  getTestLinkMRR(bool type_constrain = false) {
    if (type_constrain)
        return mrrTC;    
    return mrr;
}


/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList = NULL;

extern "C"
void getNegTest() {
    if (negTestList == NULL)
        negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        if (randd(0) % 1000 < 500)
            negTestList[i].t = corrupt_head(0, testList[i].h, testList[i].r);
        else
            negTestList[i].h = corrupt_tail(0, testList[i].t, testList[i].r);
    }
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}
#endif

/*
MIT License

Copyright (c) 2020 Rashid Lafraie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef VALID_H
#define VALID_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

INT lastValidHead = 0;
INT lastValidTail = 0;
	
REAL l_valid_filter_tot = 0;
REAL r_valid_filter_tot = 0;

extern "C"
void validInit() {
    printf("Initialize validation\n");
    lastValidHead = 0;
    lastValidTail = 0;
    l_valid_filter_tot = 0;
    r_valid_filter_tot = 0;
}

extern "C"
void getValidHeadBatch(INT *ph, INT *pt, INT *pr) {
    INT validH = validList[lastValidHead].h; 
    INT validT = validList[lastValidHead].t;
    INT validR = validList[lastValidHead].r;
    INT offset = -1;

    ph[0] = validH;
    pt[0] = validT;
    pr[0] = validR;

    if(incrementalSetting){
        for(INT i=1; i<num_currently_contained_entities;i++) {
            if(currently_contained_entities[i + offset] == validH)
                offset++;
            
            INT entity = currently_contained_entities[i + offset]; 

            ph[i] = entity;
            pt[i] = validT;
            pr[i] = validR;
        } 
    }else{
        for (INT i = 1; i < entityTotal; i++) {
            if (i + offset == validH)
                offset++;
            
            ph[i] = i + offset;
            pt[i] = validT;
            pr[i] = validR;
        }
    }
    lastValidHead++;
}

extern "C"
void getValidTailBatch(INT *ph, INT *pt, INT *pr) {
    INT validH = validList[lastValidTail].h; 
    INT validT = validList[lastValidTail].t;
    INT validR = validList[lastValidTail].r;
    INT offset = -1;

    ph[0] = validH;
    pt[0] = validT;
    pr[0] = validR;

    if(incrementalSetting){
        for(INT i=1; i<num_currently_contained_entities;i++) {
            if(currently_contained_entities[i + offset] == validT)
                offset++;
            
            INT entity = currently_contained_entities[i + offset]; 

            ph[i] = validH;
            pt[i] = entity;
            pr[i] = validR;
        } 
    }else{
        for (INT i = 1; i < entityTotal; i++) {
            if (i + offset == validT)
                offset++;
            
            ph[i] = validH;
            pt[i] = i + offset;
            pr[i] = validR;
        }
    }
    lastValidTail++;
}

extern "C"
void validHead(REAL *con, INT lastValidHead) {
    INT h = validList[lastValidHead].h;
    INT t = validList[lastValidHead].t;
    INT r = validList[lastValidHead].r;
    INT filter_offset = -1;

    REAL minimal = con[0];
    INT l_filter_s = 0;
    
    if (minimal != INFINITY){
        if (incrementalSetting){
            for (INT j = 1; j < num_currently_contained_entities; j++) {
                REAL value = con[j];
                if (currently_contained_entities[j + filter_offset] == h)
                        filter_offset++;
                    
                if (value < minimal) {
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
                    if (not _find(j + filter_offset, t, r)){
                        l_filter_s += 1;
                    }
                }
            }
        }
    } else {
        if (incrementalSetting){
            l_filter_s = num_currently_contained_entities;

            for (INT j = 1; j < num_currently_contained_entities; j++) {
                if (currently_contained_entities[j + filter_offset] == h)
                    filter_offset++;
                                
                if (_find(currently_contained_entities[j + filter_offset], t, r)){
                    l_filter_s -= 1;
                }
            }
        } else {
            l_filter_s = entityTotal;
            
            for (INT j = 1; j < entityTotal; j++){ 
                if (j + filter_offset == h)
                        filter_offset++;
                
                if (_find(j + filter_offset, t, r))
                    l_filter_s -= 1;
                
            }
        }
    }
    if (l_filter_s < 10) l_valid_filter_tot += 1;

}

extern "C"
void validTail(REAL *con, INT lastValidTail) {
    INT h = validList[lastValidTail].h;
    INT t = validList[lastValidTail].t;
    INT r = validList[lastValidTail].r;
    INT filter_offset = -1;

    REAL minimal = con[0];
    INT r_filter_s = 0;
    
    if (minimal != INFINITY){
        if (incrementalSetting){
            for (INT j = 1; j < num_currently_contained_entities; j++) {
                REAL value = con[j];
                if (currently_contained_entities[j + filter_offset] == t)
                        filter_offset++;
                    
                if (value < minimal) {
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
                    if (not _find(h, j + filter_offset, r)){
                        r_filter_s += 1;
                    }
                }
            }
        }
    } else {
        if (incrementalSetting){
            r_filter_s = num_currently_contained_entities;    

            for (INT j = 1; j < num_currently_contained_entities; j++) {
                if (currently_contained_entities[j + filter_offset] == t)
                    filter_offset++;
                
                if (_find(h, currently_contained_entities[j + filter_offset], r))
                    r_filter_s -= 1;
            }
        } else {
            r_filter_s = entityTotal;
            for (INT j = 1; j < entityTotal; j++){ 
                if (j + filter_offset == t)
                        filter_offset++;
                
                if (_find(h, j + filter_offset, r))
                    r_filter_s -= 1;
            }
        }
    }
    if (r_filter_s < 10) r_valid_filter_tot += 1;
}

REAL validHit10 = 0;
extern "C"
REAL  getValidHit10() {
    printf("Valid Total is: %ld.\n", validTotal);
    printf("Triple Total is: %ld.\n", tripleTotal);
    if (incrementalSetting)
        printf("Currently contained entities: %ld.\n", num_currently_contained_entities);
    else
        printf("Entity Total: %ld.\n", entityTotal);
    
    l_valid_filter_tot /= validTotal;
    r_valid_filter_tot /= validTotal;
    validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2;
   // printf("result: %f\n", validHit10);
    return validHit10;
}

#endif
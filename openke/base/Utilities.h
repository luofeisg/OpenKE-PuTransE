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

#ifndef UTILITIES_H
#define UTILITIES_H
#include "Utilities.h"
#include "Triple.h"
/*
================== Utility functions ==========================
*/

bool isEqual(const char *string1, const char *string2) {
    return strcmp(string1, string2) == 0;
}

std::string int_to_string(int num){
    std::string str = "";
    char int_buffer[12];
    sprintf(int_buffer, "%d", num);
    for (INT i = 0; i < 1; i++)
		 str = str + int_buffer[i];
    
    return str;
}

INT getLineNum(FILE *fl){
    int line_count = 0;
    char c;
  
    for (c = getc(fl); c != EOF; c = getc(fl)) 
        if (c == '\n')
            line_count++; 
  
    rewind(fl);
    return line_count;
}


void callocIntArray(INT* &arr, INT length) {
    if(arr == NULL)
        arr = (INT *) calloc(length, sizeof(INT));
    
    else if(arr != NULL)
        arr = (INT *) realloc(arr, length * sizeof(INT));
    
    if (!arr) {
        printf("out of memory!!\n");
        exit(EXIT_FAILURE);
    }
}

void callocTripleArray(Triple* &arr, INT length) {
    if(arr == NULL)
        arr = (Triple *) calloc(length, sizeof(Triple));
        
    else if(arr != NULL)
        arr = (Triple *) realloc(arr, length * sizeof(Triple));
    
    if (!arr) {
        printf("out of mem\n");
        exit(EXIT_FAILURE);
    }
}

void callocRealArray(REAL* &arr, INT length) {
    if(arr == NULL){
        arr = (REAL *) calloc(length, sizeof(REAL));
        // printf("REAL array allocated.\n");
    }
    
    if (!arr) {
        printf("out of mem\n");
        exit(EXIT_FAILURE);
    }
}

void resetTripleHelper(Triple *&helper) {
    if (helper != NULL) {
        free(helper);
        helper = NULL;
    }
}

void resetIntHelper(INT *&helper) {
    if (helper != NULL) {
        free(helper);
        helper = NULL;
    }
}

void resetRealHelper(REAL *&helper) {
    if (helper != NULL) {
        free(helper);
        helper = NULL;
    }
}

#endif

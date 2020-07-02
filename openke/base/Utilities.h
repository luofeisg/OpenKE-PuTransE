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
    char int_buffer[1];
    sprintf(int_buffer, "%d", num);
    for (INT i = 0; i < strlen(int_buffer); i++)
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
        printf("out of mem\n");
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

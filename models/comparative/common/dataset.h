#ifndef AROMARL_DATASET_H
#define AROMARL_DATASET_H

#include <stddef.h>

#define AROMARL_MAX_LABEL_LEN 32
#define AROMARL_MAX_CLASSES 32
#define AROMARL_MAX_FEATURES 64

typedef struct{
    int n_samples;
    int n_features;
    int n_classes;


    float *x;   //x[i*n_features + j]

    int *y; //y[i] in [0, n_classes)

    charclass_names[AROMARL_MAX_CLASSES][AROMARL_MAX_LABEL_LEN];
} Dataset;

//Load a CSV from `path`
int dataset_load_csv(const char *path,Dataset *out);

//free the heap
void dataset_free(Dataset *ds);

// Random 80/20 split
void dataset_split(const Dataset *src,int seed,Dataset *train,Dataset *test);


void dataset_feature_stats(const Dataset *ds,float *out_mean,float *out_std);

//x' = (x - mean) / std
void dataset_standardize(Dataset *ds,const float *mean,const float *std);

#endif /* AROMARL_DATASET_H */

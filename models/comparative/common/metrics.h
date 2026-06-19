#ifndef AROMARL_METRICS_H
#define AROMARL_METRICS_H

#include "dataset.h"

typedef struct {
    int cm[AROMARL_MAX_CLASSES][AROMARL_MAX_CLASSES];
    double precision[AROMARL_MAX_CLASSES];
    double recall [AROMARL_MAX_CLASSES];
    double f1 [AROMARL_MAX_CLASSES];
    int support [AROMARL_MAX_CLASSES];
    double accuracy;
} Metrics;


void metrics_compute(const int *y_true,const int *y_pred,int n, int n_classes,Metrics *out);

void metrics_print(const Metrics *m, const char *class_names[][AROMARL_MAX_LABEL_LEN],int n_classes,FILE *fp);

#endif

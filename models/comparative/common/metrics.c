#include "metrics.h"

#include <stdio.h>
#include <math.h>

void metrics_compute(const int *y_true,const int *y_pred,int n,int n_classes,Metrics *out){
    memset(out,0,sizeof(*out));

    for(int i = 0; i < n; ++i){
        int t = y_true[i], p = y_pred[i];

        if(t >= 0 && t<n_classes &&p>= 0 && p<n_classes){
            out->cm[t][p]++;
            out->support[t]++;
        }
    }

    int correct=0;
    for(int t=0;t<n_classes;++t){
        int tp= out->cm[t][t];
        int fp=0, fn = 0;

        for(int k = 0; k < n_classes; ++k){
            if(k != t){
                fp += out->cm[k][t]; fn += out->cm[t][k];
            }
        }

        correct += tp;
        out->precision[t] = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
        out->recall   [t] = (tp + fn) ? (double)tp / (tp + fn) : 0.0;

        out->f1[t]=(out->precision[t] +out->recall[t])>0.0
        ?2.0 * out->precision[t] * out->recall[t]/(out->precision[t]+out->recall[t])
        :0.0;
    }
    out->accuracy = n?(double)correct/n:0.0;
}

void metrics_print(const Metrics *m,const char *class_names[][AROMARL_MAX_LABEL_LEN],int n_classes,FILE *fp){
    fprintf(fp,"  accuracy = %.4f\n", m->accuracy);
    fprintf(fp,"  per-class:\n");
    fprintf(fp,"    %-16s %10s %10s %10s %10s\n","class", "precision", "recall", "f1", "support");


    for(int k = 0; k < n_classes; ++k){
        fprintf(fp,"    %-16s %10.4f %10.4f %10.4f %10d\n",(*class_names)[k],m->precision[k],m->recall[k], m->f1[k],m->support[k]);
    }
}

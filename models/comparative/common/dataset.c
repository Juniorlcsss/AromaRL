#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


static char* trim(char *s){
    while(*s&&isspace((unsigned char)*s)){
        s++;
    }

    char *end =s+strlen(s);
    while(end > s&&isspace((unsigned char)end[-1])){
        *--end = '\0';
    }

    return s;
}

//label string -> class index
static int label_to_class(Dataset *ds, const char *name){
    for(int i=0;i<ds->n_classes; ++i){
        if(strcmp(ds->class_names[i], name) ==0){
            return i;
        }
    }

    if(ds->n_classes >=AROMARL_MAX_CLASSES){
        fprintf(stderr, "dataset: too many classes (>%d)\n",AROMARL_MAX_CLASSES);
        return -1;
    }

    int i = ds->n_classes++;
    snprintf(ds->class_names[i], AROMARL_MAX_LABEL_LEN, "%s", name);
    return i;
}

//get column count
static int count_columns(char *line){
    int n=1;
    for(char *p =line; *p;++p){
        if (*p ==','){
            n++;
        }
    }

    while(n > 1&&line[strlen(line)-1] ==','){
        n--;
    }
    return n;
}

int dataset_load_csv(const char *path, Dataset *out){
    memset(out, 0, sizeof(*out));

    FILE *fp = fopen(path, "r");
    if(!fp){
        perror(path); return -1;
    }

    char line[1024];


    //header
    if(!fgets(line, sizeof(line), fp)){
        fprintf(stderr,"dataset: empty file %s\n",path);
        fclose(fp);
        return -1;
    }

    char *header=trim(line);
    int n_cols=count_columns(header);
    if(n_cols<3){
        fprintf(stderr,"dataset: need idx,label,f0,... (got %d cols)\n", n_cols);
        fclose(fp);
        return -1;
    }

    out->n_features= n_cols - 2;
    if(out->n_features > AROMARL_MAX_FEATURES){
        fprintf(stderr, "dataset: too many features (>%d)\n", AROMARL_MAX_FEATURES);
        fclose(fp);
        return -1;
    }

    //!rows
    /* Two-pass would be cleaner, but we grow the buffer on demand. */
    size_t cap = 256;
    out->x = (float *)malloc(cap * out->n_features * sizeof(float));
    out->y = (int *)malloc(cap * sizeof(int));

    if(!out->x || !out->y){
        perror("malloc");
        fclose(fp);
        return -1;
    }

    while(fgets(line, sizeof(line), fp)){
        char *row=trim(line);

        if(*row =='\0'||*row =='#'){
            continue;
        }

        char *fields[AROMARL_MAX_FEATURES +2];
        int nf=0;
        char *save=NULL;
        char *tok = strtok_r(row, ",", &save);

        while(tok&&nf < n_cols){
            fields[nf++] = trim(tok);
            tok = strtok_r(NULL, ",", &save);
        }

        if(nf < n_cols){
            continue;//skip incomplete rows
        }

        int cls = label_to_class(out,  fields[1]);
        if(cls<0){
            fclose(fp); return -1;
        }

        if((size_t)out->n_samples >= cap){
            cap *= 2;
            float *nx = (float *)realloc(out->x, cap * out->n_features * sizeof(float));
            int *ny = (int *)realloc(out->y, cap * sizeof(int));

            if(!nx||!ny){
                perror("realloc"); fclose(fp); return -1;
            }
            out->x = nx; out->y = ny;
        }

        float *xr = &out->x[out->n_samples * out->n_features];

        for(int j=0; j <out->n_features;++j){
            xr[j]=strtof(fields[2 + j], NULL);
        }
        out->y[out->n_samples] = cls;
        out->n_samples++;
    }
    fclose(fp);
    return 0;
}

void dataset_free(Dataset *ds){
    if (!ds) return;
    free(ds->x); free(ds->y);
    ds->x = NULL; ds->y = NULL;
    ds->n_samples = ds->n_features = ds->n_classes = 0;
}

void dataset_split(const Dataset *src, int seed,Dataset *train, Dataset *test){
    //fisher yates with a deterministic LCG when seed >= 0
    unsigned long s = (seed >= 0)?(unsigned long)seed : 0xC0FFEEul;
    int *order = (int *)malloc(src->n_samples * sizeof(int));

    for(int i =0; i < src->n_samples; ++i){
        order[i]=i;
    }

    for(int i=src->n_samples - 1; i > 0; --i){
        s = 1664525ul * s + 1013904223ul;
        int j =(int)(s % (unsigned long)(i + 1));
        int tmp =order[i];

        order[i] = order[j];
        order[j] = tmp;
    }

    int n_train= (src->n_samples * 80) / 100;
    memset(train,0, sizeof(*train));
    memset(test,0,sizeof(*test));

    train->n_features= test->n_features = src->n_features;
    train->n_classes =test->n_classes  = src->n_classes;

    memcpy(train->class_names, src->class_names, sizeof(src->class_names));
    memcpy(test ->class_names,src->class_names, sizeof(src->class_names));

    train->x = (float *)malloc(n_train * src->n_features * sizeof(float));
    train->y = (int *)malloc(n_train * sizeof(int));

    int n_test = src->n_samples - n_train;

    test->x =(float *)malloc(n_test * src->n_features * sizeof(float));
    test->y=(int *)malloc(n_test * sizeof(int));

    for(int k = 0; k < n_train; ++k){
        memcpy(&train->x[k * src->n_features],&src->x[order[k] * src->n_features],src->n_features * sizeof(float));

        train->y[k]=src->y[order[k]];
    }

    for(int k=0; k <n_test; ++k){
        memcpy(&test->x[k * src->n_features],&src->x[order[n_train + k] * src->n_features],src->n_features * sizeof(float));

        test->y[k] =src->y[order[n_train + k]];
    }
    train->n_samples = n_train;
    test->n_samples = n_test;
    free(order);
}

void dataset_feature_stats(const Dataset *ds,float *out_mean,float *out_std){
    for(int j = 0; j < ds->n_features; ++j){
        double sum = 0.0, sum2 = 0.0;

        for(int i = 0; i < ds->n_samples; ++i){
            float v =ds->x[i * ds->n_features+ j];
            sum+=v;
            sum2+=(double)v * v;
        }

        double m=sum/ds->n_samples;
        double v=sum2/ds->n_samples- m * m;

        if (v < 0.0){
            v = 0.0;
        }

        out_mean[j]=(float)m;
        out_std [j]=(float)((v > 0.0) ? sqrt(v) : 1.0);

        if(out_std[j] < 1e-6f){
            out_std[j] = 1e-6f;
        }
    }
}

void dataset_standardize(Dataset *ds,const float *mean, const float *std){
    for(int i = 0; i < ds->n_samples; ++i){
        for(int j = 0; j < ds->n_features; ++j){
            ds->x[i * ds->n_features + j]=(ds->x[i * ds->n_features + j] - mean[j])/std[j];
        }
    }
}

#ifndef AROMARL_UTIL_H
#define AROMARL_UTIL_H

static inline int argmax(const float *v, int n){
    int best = 0;
    float best_v=v[0];

    for(int i=1;i<n;++i){
        if(v[i]>best_v){
            best_v=v[i]; 
            best = i;
        }
    }
    return best;
}

static inline int argmax_abs(const float *v,int n){
    int best=0;
    float best_v = v[0]<0 ? -v[0]:v[0];
    for(int i = 1; i < n; ++i){
        float a = v[i] < 0 ? -v[i]:v[i];

        if(a > best_v){
            best_v = a; best = i;
        }
    }
    return best;
}

#endif 


#include "rebound.h"
void heartbeat(struct reb_simulation* r){
	int N = r->N;
	for (int i=N-1;i>=r->N_active;i+=-1){
        double rh = r->particles[i].x*r->particles[i].x;
        rh+= r->particles[i].y*r->particles[i].y;
        rh+=r->particles[i].z*r->particles[i].z;
        rh = sqrt(rh);
        if(rh > 1500 || rh < 20.0){
            reb_remove(r, i, 1);
	        FILE* of = fopen("removal-log.txt","a"); 
            fprintf(of,"removing particle %d at time %e\n",i, r->t);
            fclose(of);
        }
    }
}

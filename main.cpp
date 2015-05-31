#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cassert>
#include <climits>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <algorithm>
using namespace std;

#include "PoissonDiskSampler.hpp"
#include "Timer.hpp"

void runExp(const size_t& w, const size_t& h, const size_t& nd,
            const float& r, FILE* logfile){
  PoissonDiskSampler* oglr = new PoissonDiskSampler(w,h,nd,r);
  size_t usedmem = oglr->init();
  
  size_t emptypixels = 0;
  size_t itr=0;
  Timer timer;
  Timer t1,t2,t3;

  for(int i=0; i < 1; i++){
    double p1=0,p2=0,p3=0;
    oglr->reset();
    glFinish();
    itr=0;
    timer.start();
    do{
      t1.start();
      oglr->throwDarts();
      glFinish();
      p1+=t1.stop();
      t2.start();
      oglr->removeConflict();
      glFinish();
      p2+=t2.stop();
      t3.start();
      emptypixels=oglr->collectEmptyPixels();
      glFinish();
      p3+=t3.stop();
      itr++;    
    }
    while(emptypixels > 0 && itr < 200 );
    double elapsed = timer.stop();
    
    //get the results
    vector<GLfloat> res;
    oglr->downloadResults(res);
    size_t npts = res.size()/2;
  
    if (logfile != NULL){
      // fprintf(logfile,"%ld\t%ld\t%ld\t%f\t%ld\t%ld\t%f\t%f\t%f\t%f\t%f\t%f\n",
      //         w,h,nd,r,npts,itr,p1*1000,p2*1000,p3*1000,
      //         elapsed*1000,usedmem/1048576.0,npts/elapsed);
      cout << npts/elapsed << "pts / sec" << endl;
    }
  
  // //savefile
  // stringstream fs;
  // fs << "filename.rps";
  // cout << fs.str() << endl;
  // FILE* outfile=fopen(fs.str().c_str(),"wb");
  // size_t s = fwrite(&res[0], sizeof(res[0]), res.size(),outfile);
  // assert (s == res.size());
  // fclose(outfile);
  }
  delete oglr;
}

float computeR(const size_t& n){
  return 0.7766*sqrt(2.0/(sqrt(3.0)*n));
}

size_t computeN(const float& r){
  return 2.0/(sqrt(3.0)*pow(r/0.7766,2));
}

int main(int argc, char** argv){
  glutInit(&argc,argv);
  glutCreateWindow (""); //create the context
  glewInit();

  float r = 8.5/4096;
  runExp(4096,4096,computeN(r)/2,r,stdout);

  return 0;
}

#include "cudaThrustOGL.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>

#include <cassert>

typedef unsigned int uint;
typedef GLubyte mask_t;

//texture binding point for cuda access (for depth map)
texture<uchar1, cudaTextureType2D, cudaReadModeElementType> cudaTex;
__constant__  GLuint texwidth;

cudaThrustOGL::cudaThrustOGL(){
  err_=cudaDeviceReset();
  err_=cudaGLSetGLDevice(0);
  err_=cudaSetDevice(0);
  assert(err_==cudaSuccess);
}

void cudaThrustOGL::cudaInit(const GLuint& texID,
                             const GLuint& bufID,
                             const GLuint& emptybufID,
                             const GLuint& resultsBufID,
                             const size_t& w, const size_t& h){
  width_ = w;
  height_ = h;
  
  //cuda register GL resources
  err_=cudaGraphicsGLRegisterImage(&cuda_res_[0],texID,
                                   GL_TEXTURE_2D,
                                   cudaGraphicsMapFlagsReadOnly);
  err_=cudaGraphicsGLRegisterBuffer(&cuda_res_[1],bufID,
                                    cudaGraphicsMapFlagsNone);
  err_=cudaGraphicsGLRegisterBuffer(&cuda_res_[2],emptybufID,
                                    cudaGraphicsMapFlagsNone);
  err_=cudaGraphicsGLRegisterBuffer(&cuda_res_[3],resultsBufID,
                                    cudaGraphicsMapFlagsNone);

  //upload the texture width to device
  GLuint uintw=(GLuint)width_;
  err_=cudaMemcpyToSymbol("texwidth", &uintw,sizeof(uintw));
    
  seed_ = (unsigned int) time(NULL);//12345

  reset();
  assert(err_==cudaSuccess);
}

void cudaThrustOGL::reset(){
  //init number of remaining darts to the size of the texture
  rem_darts_ = width_*height_; 
  iter_=0;
  rngoffset_=0; //keep using the random numbers
}

//Operator structs for counting empty pixels
struct isEmpty{
  __device__
  bool operator()(const GLuint& i){
    GLuint x = i % texwidth;
    GLuint y = i / texwidth;
    return (tex2D(cudaTex,x,y).x == 0);
  }
};

struct notEmpty{
  __device__
  bool operator()(const GLuint& i){
    GLuint x = i % texwidth;
    GLuint y = i / texwidth;
    return (tex2D(cudaTex,x,y).x != 0);
  }
};

size_t cudaThrustOGL::thrustCountEmptyPixels(){
  err_=cudaGraphicsMapResources(3,&cuda_res_[0]);

  //get the texture array
  cudaArray* cuda_array;
  err_=cudaGraphicsSubResourceGetMappedArray(&cuda_array,
                                             cuda_res_[0],0,0);
  //bind the texture to cuda
  err_=cudaBindTextureToArray(cudaTex, cuda_array);

  //get the emptylist buffer
  GLuint* emptylistbuf;
  size_t bufsize;
  err_=cudaGraphicsResourceGetMappedPointer((void**)&emptylistbuf,&bufsize,
                                            cuda_res_[2]);
  
  //convert raw ptr to thrust ptr
  thrust::device_ptr<GLuint> dev_ptr
      =thrust::device_pointer_cast(emptylistbuf);

  //cout << "before thrust " << dev_ptr[0] << " " << dev_ptr[1] << endl;  

  //declare the newend of the emptylist
  thrust::device_ptr<GLuint> newend;

  if(iter_==0){ // use counting itr to accelerate the first iteration
    newend = thrust::copy_if(thrust::make_counting_iterator<GLuint>(0),
                             thrust::make_counting_iterator<GLuint>(
                                 rem_darts_),dev_ptr, isEmpty());
  }
  else{
    newend = thrust::remove_if(dev_ptr,dev_ptr+rem_darts_ ,notEmpty());   
  }
  
  //cout << "after thrust " << dev_ptr[0] << " " << dev_ptr[1] << endl;

  err_=cudaUnbindTexture(cudaTex);
  err_=cudaGraphicsUnmapResources(3,&cuda_res_[0]);

  size_t newrem_darts=newend-dev_ptr;  
  assert(newrem_darts < rem_darts_);

  rem_darts_ = newrem_darts;

  iter_++;
  assert(err_==cudaSuccess);
  return rem_darts_;
}

void cudaThrustOGL::cudaCleanup(){
  cudaGraphicsUnmapResources(3,&cuda_res_[0]);  
  for(size_t i=0; i< 3; i++){
    cudaGraphicsUnregisterResource(cuda_res_[i]);
  }
  //cudaFree(drandvec);
}

size_t cudaThrustOGL::freeGPUMem(){
  glFinish();
  err_=cudaDeviceSynchronize();
  size_t avail;
  size_t total;
  err_=cudaMemGetInfo( &avail, &total );
  //cout << "Device memory available: " << avail*1.0/1048576 << "MB" <<endl;
  assert(err_==cudaSuccess);
  return avail;
}


void cudaThrustOGL::remDuplicateSamples(size_t dartCount){
  unsigned int *buf;
  size_t bufSize;
  err_=cudaGraphicsMapResources(1,&cuda_res_[3]);
  err_=cudaGraphicsResourceGetMappedPointer((void **)&buf,
                                            &bufSize, cuda_res_[3]);

  thrust::device_ptr<unsigned int> buf_first(buf);
  thrust::device_ptr<unsigned int> buf_last = buf_first+dartCount;
  thrust::device_ptr<unsigned int> buf_last_new
      = thrust::unique(buf_first,buf_last);

  cudaGraphicsUnmapResources(1,&cuda_res_[3]);
}


//Random uniform distribution for transform
template <typename T> class random_uniform{
 private:
  thrust::random::default_random_engine rng;
  //thrust::random::ranlux48 rng;
  thrust::uniform_real_distribution<float> dist;  
  //Pointer to the empty list
  const T* empty_ptr_;//,*r_ptr_;

  //Offset for the rng
  const size_t offset_;

  //Number of elements in the empty list
  T umax_;

  //Size of the texture
  const size_t w_, h_;

  //scale normalized coord to ushort variables
  float wscale_,hscale_;
  
  //subpixel variables
  float sppixelw_,sprowarea_;
  
 public:
  // provide constructor to initialize the distribution:
  random_uniform(const size_t& w, const size_t& h, const T& umax,
                 const T* dp, const size_t& o, const unsigned int& s)
      :dist(),empty_ptr_(dp), offset_(o),umax_(umax),w_(w),h_(h){
    rng.seed(s);

    hscale_ = (USHRT_MAX*1.0f/h_);
    wscale_ = (USHRT_MAX*1.0f/w_);

    sppixelw_ = h_*1.0f/USHRT_MAX * w_*1.0f/USHRT_MAX;
    sprowarea_ = h_*1.0f/USHRT_MAX;
  }

  // OK, now the actual operator:
  __device__
  T operator()(size_t index){
    // skip past numbers used in previous threads
    rng.discard(index+offset_); //offset the used random numbers

    float fidx = dist(rng)*umax_;
    T coord =floor(fidx);
    float frac=fidx-floor(fidx);

    if(empty_ptr_ != NULL){
      coord = empty_ptr_[coord]; //sample from the empty list
    }

    //pack the coordinates into ushorts
    GLushort x = (coord % w_)* wscale_;
    GLushort y = (coord / w_)* hscale_;

    //subpixel
    x += floor(fmod(frac, sprowarea_)/sppixelw_); // div subpix width
    y += floor(frac / sprowarea_); //div subpix row area
    
    //pack x y into uint
    T res = (y << 16 | x & 0xffff);
    return res;
  }
};

//Generate some vertices
void cudaThrustOGL::makeVertices(const size_t& ndarts){
  err_=cudaGraphicsMapResources(3,&cuda_res_[0]);
   
  //get the dart buffer
  GLuint* dartbuf;
  size_t bufsize;
  err_=cudaGraphicsResourceGetMappedPointer((void**)&dartbuf,&bufsize,
                                            cuda_res_[1]);

  //get the emptylist buffer
  GLuint* emptylistbuf;
  err_=cudaGraphicsResourceGetMappedPointer((void**)&emptylistbuf,&bufsize,
                                            cuda_res_[2]);

  //convert raw ptr to thrust ptr
  thrust::device_ptr<GLuint> dart_ptr=thrust::device_pointer_cast(dartbuf);

  if(iter_==0){
    emptylistbuf=NULL; //do not use the emptylist lookup in iter 0
  }

  //pick ndarts locations from the emptylist
  thrust::transform(thrust::make_counting_iterator<GLuint>(0),
                    thrust::make_counting_iterator<GLuint>(ndarts),
                    dart_ptr,random_uniform<GLuint>(width_,height_,
                                                    rem_darts_-1,
                                                    emptylistbuf,
                                                    rngoffset_,seed_));
  rngoffset_+=ndarts;
  err_=cudaGraphicsUnmapResources(3,&cuda_res_[0]);
  assert(err_==cudaSuccess);
}

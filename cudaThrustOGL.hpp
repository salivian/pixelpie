#ifndef __CUDATHRUSTOGL__
#define __CUDATHRUSTOGL__

#if defined _WIN32 || defined _WIN64
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

class cudaThrustOGL{
 private:
  // 0: coverage texture
  // 1: dart source buffer
  // 2: empty pixel list
  // 3: result sample buffer
  cudaGraphicsResource_t cuda_res_[4];

  size_t width_,height_;  
  size_t rem_darts_;
  size_t iter_;
  size_t rngoffset_;
  unsigned int seed_;
  cudaError_t err_;

 public:
  cudaThrustOGL();
  ~cudaThrustOGL(){cudaCleanup();};

  void cudaInit(const GLuint& texID, const GLuint& bufID,
		const GLuint& emptybufID,
                const GLuint& resultsBufID,
		const size_t& w, const size_t& h);
  void cudaCleanup();
  void reset();

  void makeVertices(const size_t& ndarts);

  size_t thrustCountEmptyPixels();
  size_t getRemainingDarts() const {return rem_darts_;}
  size_t freeGPUMem();

  void remDuplicateSamples(size_t dartCount);
};

#endif

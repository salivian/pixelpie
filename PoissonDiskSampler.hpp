#ifndef __POISSONDISKSAMPLER__
#define __POISSONDISKSAMPLER__

#include <GL/glew.h>

#include <vector>
#include <string>
using namespace std;

#include <cudaThrustOGL.hpp>

class PoissonDiskSampler{
 public:
  PoissonDiskSampler(const size_t& w, const size_t& h,
                     const size_t& nd, const float& rd);
  ~PoissonDiskSampler();

  size_t init();
  void reset(); 

  void cleanup();

  // Pass 1: Dart throwing Step
  void throwDarts();
  // Pass 2: Conflict removal Step
  void removeConflict();
  // Post-Pass: empty pixel removal/compaction
  size_t collectEmptyPixels();

  void saveImage(const string& filename) const;
  void saveEmptyList(const string& filename) const;
  void downloadResults(std::vector<GLfloat>& res);

  //Load an importance map and activate the importance texture
  void loadImportanceMap(const string& filename);

 private:
  size_t width_,height_,ndarts_;
  const size_t ond_;
  float dartradius_;

  GLuint res_offset_;
  GLuint resultsbuffer_size_;
  std::vector<GLshort> random_vertices_;  

  // OpenGL programs
  GLuint programThrow_;
  GLuint programRemove_;
  GLuint LoadShaders(const string& vertex_file_path,
                     const string& geometry_file_path,
                     const string& fragment_file_path);
  void initPrograms();

  // OpenGL buffers
  GLuint sourceBuffer_;
  GLuint resultsBuffer_;
  GLuint emptylistBuffer_;

  // OpenGL Frame buffer
  GLuint frameBuffer_;
  void initFBO();

  // OpenGL textures
  GLuint depthTexture_;
  GLuint coverageTexture_;
  GLuint importancetex_;

  // Cuda implementation wrapper
  cudaThrustOGL* cuda_thrust_ogl_obj_;

  GLuint query_;
  GLuint VertexArrayID_;
};

#endif

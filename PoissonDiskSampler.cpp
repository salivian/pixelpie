
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <string>

using namespace std;

#include "PoissonDiskSampler.hpp"

#include <cuda_runtime.h>
#include "lodepng.h"

#include "Timer.hpp"

#define MINDARTS 1024

PoissonDiskSampler::PoissonDiskSampler(const size_t& w, const size_t& h,
                                       const size_t& nd, const float& rd)
    :width_(w),height_(h),ndarts_(nd),ond_(nd),dartradius_(rd),res_offset_(0){
  assert(width_ > 0);
  assert(height_ > 0);
  assert(ndarts_ > 0);
  assert(dartradius_ > 0);
}

PoissonDiskSampler::~PoissonDiskSampler(){
  cleanup();
}

//Initial the FBO and textures for dart throwing and conflict checking
void PoissonDiskSampler::initFBO(){
  //Create Textures
  glGenTextures(1, &depthTexture_);
  glGenTextures(1, &coverageTexture_);

  // Setup 24bit depth texture
  glBindTexture(GL_TEXTURE_2D, depthTexture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width_, height_, 0,
               GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);

  //Activate depth comparison for pass 2
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE,
                  GL_COMPARE_R_TO_TEXTURE);

  //Reject the dart if its depth is greater than the depth map.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC,GL_GREATER);

  // Setup 8bit coverage texture
  glBindTexture(GL_TEXTURE_2D, coverageTexture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, width_, height_, 0,
               GL_RED_INTEGER, GL_UNSIGNED_BYTE, 0);

  // Setup framebuffer
  glGenFramebuffers(1, &frameBuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer_);

  //Depth map
  glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,
                         GL_TEXTURE_2D,depthTexture_,0);

  //Coverage map
  glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
                         GL_TEXTURE_2D,coverageTexture_,0);
  GLenum status;
  status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  assert(status==GL_FRAMEBUFFER_COMPLETE);
}

void PoissonDiskSampler::initPrograms(){
  // Create and compile our GLSL program from the shaders
  programThrow_ = LoadShaders( "VertexShader.vs",
                               "DartThrowing1.gs",
                               "FragmentShader1.fs" );

  programRemove_ = LoadShaders( "VertexShader.vs",
                                "ConflictRemoval2.gs",
                                "FragmentShader2.fs" );

  // Setup transform feedback at geometry shader of 2nd pass
  GLchar const * Strings[] = {"feedbackPos"};
  glTransformFeedbackVaryings(programRemove_, 1, Strings,
                              GL_SEPARATE_ATTRIBS);
  glLinkProgram(programRemove_); //relink the program

  //upload the uniforms
  { // Throw pass
    glUseProgram(programThrow_);
    GLint radiusLoc = glGetUniformLocation(programThrow_,"dartradius");
    glUniform1f(radiusLoc, dartradius_);
    GLint importanceTexLoc = glGetUniformLocation(programThrow_, "impTex");
    glUniform1i(importanceTexLoc, 1);
  }

  { // Remove pass
    glUseProgram(programRemove_);
    GLint radiusLoc = glGetUniformLocation(programRemove_,"dartradius");
    glUniform1f(radiusLoc, dartradius_);
    GLint importanceTexLoc = glGetUniformLocation(programRemove_, "impTex");
    glUniform1i(importanceTexLoc, 1);
    GLint depthTexLoc = glGetUniformLocation(programRemove_, "depthTex");
    glUniform1i(depthTexLoc, 0);
  }
}


//Initialize buffers, vertex arrays, call cuda init and fbo init
size_t PoissonDiskSampler::init(){
  cuda_thrust_ogl_obj_ = new cudaThrustOGL;
  size_t oldmem = cuda_thrust_ogl_obj_->freeGPUMem();

  ndarts_ = max(ndarts_, (size_t)MINDARTS);

  // Setup dart input buffer
  glGenBuffers(1, &sourceBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, sourceBuffer_);
  glBufferData(GL_ARRAY_BUFFER,sizeof(GLuint)*ndarts_,NULL,GL_DYNAMIC_DRAW);

  // Setup empty List Buffer
  glGenBuffers(1, &emptylistBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, emptylistBuffer_);
  glBufferData(GL_ARRAY_BUFFER,sizeof(GLuint)*width_*height_,NULL,
               GL_DYNAMIC_DRAW);

  // Setup result buffer
  //120% of the estimate # of samples
  resultsbuffer_size_ = 2.0/(sqrt(3.0)*pow(dartradius_/0.7766,2))*1.2;
  glGenBuffers(1, &resultsBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, resultsBuffer_);
  glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*2*3*resultsbuffer_size_,NULL,
               GL_STATIC_DRAW);

  // Setup primitive query object
  glGenQueries(1,&query_);

  // Setup vertex data
  glGenVertexArrays(1, &VertexArrayID_);

  initFBO();

  initPrograms();

  //init Cuda
  cuda_thrust_ogl_obj_->cudaInit(coverageTexture_, sourceBuffer_,
                                 emptylistBuffer_, resultsBuffer_,
                                 width_,height_);

  reset();
  glFinish();
  return  oldmem-cuda_thrust_ogl_obj_->freeGPUMem();
}

void PoissonDiskSampler::reset(){
  glBindVertexArray(VertexArrayID_);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, sourceBuffer_);
  glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, 0, (void*)0);
  
  //clear coverage map
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glClear(GL_COLOR_BUFFER_BIT);

  glViewport(0,0,width_,height_);  

  //reset results
  res_offset_ = 0;
  //reset ndarts
  ndarts_=ond_;
  //reset cuda
  cuda_thrust_ogl_obj_->reset();
}

void PoissonDiskSampler::cleanup(){
  delete cuda_thrust_ogl_obj_;

  //FramebufferObject::Disable();
  glDeleteTextures(1,&depthTexture_);
  glDeleteTextures(1,&coverageTexture_);
  glDeleteTextures(1,&importancetex_);

  glDeleteProgram(programThrow_);
  glDeleteProgram(programRemove_);

  glDeleteFramebuffers(1,&frameBuffer_);

  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1,&VertexArrayID_);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1,&sourceBuffer_);
  glDeleteBuffers(1,&resultsBuffer_);
  glDeleteBuffers(1,&emptylistBuffer_);
  glDeleteQueries(1,&query_);
  glFinish();
}

GLuint PoissonDiskSampler::LoadShaders(const string& vertex_file_path,
                                       const string& geometry_file_path,
                                       const string& fragment_file_path){

  // Create the shaders
  GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
  GLuint GeometryShaderID = glCreateShader(GL_GEOMETRY_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  // Read the Geometry Shader code from the file
  std::ifstream VertexShaderStream(vertex_file_path.c_str(),
                                   std::ios::in);
  std::ifstream GeometryShaderStream(geometry_file_path.c_str(),
                                     std::ios::in);
  std::ifstream FragmentShaderStream(fragment_file_path.c_str(),
                                     std::ios::in);

  // Read the Vertex Shader code from the file
  std::string VertexShaderCode;
  std::string GeometryShaderCode;
  std::string FragmentShaderCode;

  if(VertexShaderStream.is_open()){
    VertexShaderCode =
        std::string((std::istreambuf_iterator<char>(VertexShaderStream)),
                    istreambuf_iterator<char>());
    VertexShaderStream.close();
  }
  if(GeometryShaderStream.is_open()){
    GeometryShaderCode =
        std::string((std::istreambuf_iterator<char>(GeometryShaderStream)),
                    istreambuf_iterator<char>());
    GeometryShaderStream.close();
  }
  if(FragmentShaderStream.is_open()){
    FragmentShaderCode =
        std::string((std::istreambuf_iterator<char>(FragmentShaderStream)),
                    istreambuf_iterator<char>());
    FragmentShaderStream.close();
  }

  GLint Result = GL_FALSE;
  int InfoLogLength;

  // Compile Vertex Shader
  const GLchar *vst=VertexShaderCode.c_str();
  glShaderSource(VertexShaderID, 1, &vst , NULL);
  glCompileShader(VertexShaderID);
  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  std::vector<char> VertexShaderErrorMessage(InfoLogLength);
  glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
                     &VertexShaderErrorMessage[0]);
  if (strlen(&VertexShaderErrorMessage[0]) > 0){
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
  }

  // Compile Geometry Shader
  const GLchar* gst=GeometryShaderCode.c_str();
  glShaderSource(GeometryShaderID, 1, &gst , NULL);
  glCompileShader(GeometryShaderID);
  // Check Geometry Shader
  glGetShaderiv(GeometryShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(GeometryShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  std::vector<char> GeometryShaderErrorMessage(InfoLogLength);
  glGetShaderInfoLog(GeometryShaderID, InfoLogLength, NULL,
                     &GeometryShaderErrorMessage[0]);
  if (strlen(&GeometryShaderErrorMessage[0]) > 0){
    fprintf(stdout, "%s\n", &GeometryShaderErrorMessage[0]);
  }

  // Compile Fragment Shader
  const GLchar* fst=FragmentShaderCode.c_str();
  glShaderSource(FragmentShaderID, 1, &fst , NULL);
  glCompileShader(FragmentShaderID);
  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
  glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
                     &FragmentShaderErrorMessage[0]);
  if (strlen(&FragmentShaderErrorMessage[0]) > 0){
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
  }

  // Link the program
  GLuint ProgramID = glCreateProgram();
  glAttachShader(ProgramID, VertexShaderID);
  glAttachShader(ProgramID, GeometryShaderID);
  glAttachShader(ProgramID, FragmentShaderID);
  glLinkProgram(ProgramID);

  // Check the program
  glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
  glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  std::vector<char> ProgramErrorMessage( max(InfoLogLength, int(1)) );
  glGetProgramInfoLog(ProgramID, InfoLogLength, NULL,
                      &ProgramErrorMessage[0]);
  if (strlen(&ProgramErrorMessage[0]) > 0){
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
  }

  return ProgramID;
}

void PoissonDiskSampler::throwDarts(){
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  glClearDepth(1.0f);
  glClear(GL_DEPTH_BUFFER_BIT);

  ndarts_ = min(ndarts_, cuda_thrust_ogl_obj_->getRemainingDarts());
  ndarts_ = max(ndarts_, (size_t)MINDARTS);

  //Generate some random darts
  cuda_thrust_ogl_obj_->makeVertices(ndarts_);

  glDrawBuffer(GL_NONE); //no render targets
  glUseProgram(programThrow_);

  glDrawArrays(GL_POINTS, 0, ndarts_);
}

void PoissonDiskSampler::removeConflict(){
  glDisable(GL_DEPTH_TEST);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, depthTexture_);

  //check if we still have space
  //(if this assert fails increase feedback buffer size)
  assert(resultsbuffer_size_-res_offset_ > 0);

  //bind buffer for capturing results
  glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                    resultsBuffer_,
                    res_offset_*2*sizeof(GLfloat)*3,
                    (resultsbuffer_size_-res_offset_)*2*sizeof(GLfloat)*3);

  glDrawBuffer(GL_COLOR_ATTACHMENT0);

  glUseProgram(programRemove_);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query_);
  glBeginTransformFeedback(GL_TRIANGLES);

  glDrawArrays(GL_POINTS, 0, ndarts_);

  glEndTransformFeedback();
  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  GLuint PrimitivesWritten = 0; //query for the # of accepted darts
  glGetQueryObjectuiv(query_, GL_QUERY_RESULT, &PrimitivesWritten);
  //cout << PrimitivesWritten << endl;
  res_offset_ += PrimitivesWritten;
}

//Count the empty pixels by call thrust
size_t  PoissonDiskSampler::collectEmptyPixels(){
  return cuda_thrust_ogl_obj_->thrustCountEmptyPixels();
}

void PoissonDiskSampler::loadImportanceMap(const string& filename){
  std::vector<unsigned char> image; //the raw pixels
  unsigned int iwidth, iheight;

  //decode
  unsigned int error = lodepng::decode(image, iwidth, iheight,
                                       std::string(filename));
  assert(error == 0);
  cout << "Importance Map: " << iwidth << " " << iheight << endl;

  glGenTextures(1,&importancetex_);

  glActiveTexture(GL_TEXTURE0+1);
  glBindTexture(GL_TEXTURE_2D, importancetex_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iwidth, iheight, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
}

//Save the depth map and coverage map to images
void PoissonDiskSampler::saveImage(const string& filename) const{
  vector<GLuint> pixels(width_*height_);
  vector<GLubyte> bpixels(width_*height_*4);

  glActiveTexture(GL_TEXTURE0+2); //used by this func only
  glBindTexture(GL_TEXTURE_2D, depthTexture_);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,
                &pixels[0]);
  glBindTexture(GL_TEXTURE_2D, 0);

  size_t emptypix = 0;
  GLuint maxe = *(max_element(pixels.begin(),pixels.end()));
  cout << "maxe " << maxe << endl;

  for(size_t i=0; i<pixels.size(); i++){
    if(pixels[i] < UINT_MAX ){
      //cout << pixels[i] << endl;
      bpixels[i*4] = pixels[i]*1.0/UINT_MAX*254*0x2fff;
    }
    else{
      bpixels[i*4] = 255;
      emptypix++;
    }
    bpixels[i*4+1]=bpixels[i*4+2]=bpixels[i*4];
    bpixels[i*4+3]=255;
  }

  lodepng::encode(std::string(filename)+"-p0.png", &bpixels[0],
                  width_, height_);

  glBindTexture(GL_TEXTURE_2D, coverageTexture_);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT,
                &pixels[0]);
  glBindTexture(GL_TEXTURE_2D, 0);

  emptypix = 0;
  for(size_t i=0; i<pixels.size(); i++){
    if(pixels[i]>0){
      bpixels[i*4] = pixels[i]*1.0/ndarts_*254;
    }
    else{
      bpixels[i*4] = 255;
      emptypix++;
    }
    bpixels[i*4+1]=bpixels[i*4+2]=bpixels[i*4];
    bpixels[i*4+3]=255;
  }

  lodepng::encode(std::string(filename)+"-p1.png", &bpixels[0],
                  width_, height_);
}

struct Vec2f{
  GLfloat x;
  GLfloat y;
  bool operator==(const Vec2f& b){
    return (x==b.x) && (y == b.y) ;
  }
};

//Get the sample from the feedback buffer
void PoissonDiskSampler::downloadResults(vector<GLfloat>& res){
  res.resize(res_offset_*2*3); //resize the results buffer

  //download the results from the feedback buffer
  glBindBuffer(GL_ARRAY_BUFFER, resultsBuffer_);
  glGetBufferSubData(GL_ARRAY_BUFFER,0,res.size()*sizeof(res[0]),&res[0]);

  //compact the duplicated pts
  Vec2f* end = unique((Vec2f*)&res[0],(Vec2f*)&res[0]+res.size()/2);
  res.resize(2*(end - (Vec2f*)&res[0]));

  //assert(res.size() == 2*res_offset_);
}

//Save the empty list into an image
void PoissonDiskSampler::saveEmptyList(const string& filename) const{
  glBindBuffer(GL_ARRAY_BUFFER, emptylistBuffer_);
  size_t listsize = cuda_thrust_ogl_obj_->getRemainingDarts();
  //cout << listsize << endl;
  vector<GLuint> res(listsize);
  glGetBufferSubData(GL_ARRAY_BUFFER,0,res.size()*sizeof(res[0]),&res[0]);

  vector<GLubyte> img(height_*width_*4);

  fill(img.begin(),img.end(), 255);
  for(size_t i=0; i < res.size();i++){
    img[res[i]*4]=0;
    img[res[i]*4+1]=255;
    img[res[i]*4+2]=0;
    img[res[i]*4+3]=255;
  }
  lodepng::encode(filename+"-e.png", &img[0], width_, height_);
}

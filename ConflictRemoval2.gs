#version 330 core
#extension GL_ARB_shading_language_packing : enable

//input
layout (points) in;
in uint dart[]; 
uniform float dartradius; 
uniform sampler2DShadow depthTex; // depth map
uniform sampler2D impTex;

//Output triangle
layout (triangle_strip,max_vertices=3) out;
out vec2 cirCoord; // Normalized circle coord of vertex
out vec2 feedbackPos; // Captured dart position

#define SQRT3 (1.7320508075688772935274463415059)

void emitTri(vec3 p, float r){
  cirCoord = vec2(0,2);
  gl_Position = vec4(p+vec3(0,2,0)*r,1); EmitVertex();
  cirCoord = vec2(-SQRT3,-1);
  gl_Position = vec4(p+vec3(-SQRT3,-1,0)*r,1); EmitVertex();
  cirCoord = vec2(SQRT3,-1);
  gl_Position = vec4(p+vec3(SQRT3,-1,0)*r,1); EmitVertex();
}
void main(){
  // Make 3D coordinate in (0,1), z is the gl_PrimitiveIDIn in 24bits
  vec3 p = vec3(unpackUnorm2x16(dart[0]),float(gl_PrimitiveIDIn)/float(0x00ffffff));

  // Check if dart is occluded(using shadow sampler and greater-than comparison mode)
  if(texture(depthTex, p) == 1.0f)
    return;

  // Capture the accepted dart by the transform feedback buffer
  feedbackPos = p.xy;

  // Sample for importance
  //float imp = texture(impTex,p.xy).x;

  // Scale p and dartradius from domain (0,1) to OpenGL domain(-1,1)
  p = p*2.0-1.0;

  // Emit triangle
  emitTri(p,dartradius*2);
}

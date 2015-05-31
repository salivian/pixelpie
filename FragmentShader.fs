#version 330 core

#extension GL_ARB_conservative_depth : enable
layout (depth_unchanged) out float gl_FragDepth;

//Input normalized coordinate;
in vec2 cirCoord;
flat in uint dartID;

// Output Color
out uint colorOut;

void main(){
  //Compute the distance from center(0,0)
  float d = dot(cirCoord,cirCoord);
  
  //Check if the fragment is outside the inscribe circle
  if(d >1.0){ 
    discard; //discard the fragment
  }

  colorOut = dartID;
}

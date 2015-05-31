#version 330 core
#extension GL_ARB_conservative_depth : enable

in vec2 cirCoord;

void main(){
  //Check if the fragment is outside the inscribe circle
  if(dot(cirCoord,cirCoord)>1.0)
    discard;
}

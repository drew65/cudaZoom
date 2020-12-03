Camfeed project
Andrew Scott
13/11/2018
drew.scott65@gmail.com
=====

Webcam

# Building on OS X

These instructions are for OS X 10.9 "Mavericks". But works with Ubuntu 16.02 +

* Requirements: OpenCV2, Cuda 8.x with NVIDIA Performance Primitives (NPP) lib.


* How to Build.
Unzip to project dir,

```
mkdir build
cd build
cmake ..
make
cd ../bin
./camfeed
```

If this process should fail then the following  will compile on Linux:
nvcc `pkg-config --cflags opencv` -o camfeed camfeed.cu `pkg-config --libs opencv`

* Keyboard commands:
```
esc : to Exit
b   : to change to greyscale
d   : counter clockwise rotation
g   : clockwise rotation
r   : zoom in (not yet functional)
v   : zoom out (not yet functional)
f   : reset to normal cam feed  
```
* The performance stats of the Cuda kernels will be displayed in the terminal window, the video will display in a separate window.
This is just a first pass at this project, a proof of concept of the underlying technology. The image transformations can be improved:
1) Visually, implementation of zoom, improvement of rotation by calculating a new output rectangle and rotation centre for each incremental rotation.  Addition of full colour to rotation.
2) Performance, the rotation currently only exists in greyscale; the grey kernel is called prior to calling the rotation, in-between these two calls the data could be left on the GPU memory.
3) Usability, the user could be given the option to define a rectangle to be transformed within the video window (via keys or mouse), then the key commands could transform the feed.

# tkDNN_python
This repository is a fork of [tkDNN](https://github.com/ceccocats/tkDNN), a Deep Neural Network library built with cuDNN and tensorRT primitives. 
It provides Python wrapping for the tkDNN object detection module.
The wrapper supports batching and three types of object detection networks (YOLO, MobileNet, and CenterNet).

## Dependencies
The usual [tkDNN dependencies](https://github.com/ceccocats/tkDNN#dependencies) are required to compile this repository. 
In addition, it requires [pybind11](https://github.com/pybind/pybind11) as well as an installation of Python (tested with 3.8.2.).
Via .gitmodules, [pybind11](https://github.com/pybind/pybind11) should be automatically downloaded to the "extern" folder.
Alternatively, it can be [manually installed](https://pybind11.readthedocs.io/en/latest/installing.html#include-as-a-submodule) using GitHub:   

```
git submodule add -b stable ../../pybind/pybind11 extern/pybind11
git submodule update --init
```
## Compilation & Installation
Follow the [compilation steps](https://github.com/ceccocats/tkDNN#how-to-compile-this-repo) provided by [tkDNN](https://github.com/ceccocats/tkDNN).

For Windows, several special steps have to be taken. 
First, follow the steps for the [Windows installation](https://github.com/ceccocats/tkDNN/blob/master/docs/windows.md) of tkDNN.
When using CMake, the include and lib directories of the dependencies (Eigen, YAML, and OpenCV) have to be specified manually.
After that, the solution and project files can be generated and compiled.
To run the programs (tests and demos) under Windows, the binary DLL folders of the dependencies have to be provided as well.
One way is to simply add them to the global $PATH$ environment variable.
A cleaner solution is to temporarily update $PATH$ before running the program.
For example, when using Visual studio, set the Debugging/Environment entry of the project file to (assuming that the dependencies have been installed to an /opt folder relative to the root folder and the build is 64-bit Release):

```
PATH=$(ProjectDir)\..\opt\x64-windows\bin;%PATH%
```

Once the tests and demos are compiled and able to run, create the network RT files.
To support batching, the [TKDNN_BATCHSIZE](https://github.com/ceccocats/tkDNN/blob/master/docs/demo.md#batching) environment variable has to be set to values greater than 1.
When using Visual Studio, that again can be achieved by extending the Debugging/Environment entry as follows:

```
TKDNN_BATCHSIZE=4
```

After compiling the pythonwrapper project, a pythonwrapper.\*.pyd file is created (\* is typically the Python version and build type, e.g., "cp38-win_amd64").
Copy the file over to the demo folder. 
Finally, ensure that all the folders for the DLL dependencies are specified in the Python demo script before importing the module (e.g., by using [add_dll_directory](https://docs.python.org/3/library/os.html) from the os module):

## Usage
Import the pythonwrapper module in the Python script (ensure that all required dependency folders are in the DLL search path).
```
import pythonwrapper
```

Create an object detector instance: 
```
network_rt_file = "../build/yolo4tiny_fp32.rt"
detector = pythonwrapper.ObjectDetector(network_rt_file)
```
You can also specify additional parameters, such as number of classes, confidence threshold, the type of network, as well as maximum batch size (if the RT file allows it).
```
network_rt_file = "../build/yolo4tiny_fp32.rt"
class_count = 80
confidence_threshold = 0.3
max_batch_size = 4 
network_type = pythonwrapper.ObjectDetector.Type.Yolo
detector = pythonwrapper.ObjectDetector(network_rt_file, class_count, confidence_threshold, max_batch_size, network_type)
```

To run inference, simply call the infer method with a list of images (an image must be a [numpy](https://numpy.org/) array with shape [height, width, 3] and dtype uint8):
```
detections = detector.infer(images)
```
The batch size is automatically determined by the number of images in the list (must not exceed the maximum batch size).
The method returns a list of detection results, one for each input image.
Each detection result is again a list of individual detections that contain the object's class ID and name, as well as bounding box and detection probability.

Note that the performance of the inference is much slower on the first call of "infer", so to estimate more precise timings, the method should be called several times.

## TODO
* DLL links have to be set manually in the Python file before importing the module which is cumbersome.
* Passing images from Python to the C++ interface currently requires a copy which is unnecessary and reduces performance. So far the impact seems small, but could be significant for larger image sizes.
* Error handling is absent. Issues in the C++ code sometimes result in error outputs, but can also just cause segfaults (e.g., when the path to the input network RT file is wrong).
* Only the Release build has been tested so far. 

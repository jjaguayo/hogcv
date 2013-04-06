Description
---------------
This small project was inspired by a paper by Dalal and Triggs, "Histograms of 
Oriented Gradients for Human Detection". The project gave me an opportunity to 
write some code using some projects and frameworks that I was interested in 
including QT, OpenCV, libsvm, CPPUnit and CMake. Most importantly, the project 
gave me an opportunity to write some code and work on my OOD skills.

The application uses a set of images to train a support vector machine that
can be used to classify a set of images as having or not having a person in 
the image.

The features for a given image are the values of a histogram of oriented 
gradients of the images pixels.

PACKAGE DEPENDENCIES
=====================
The following packages are required to build and execute the application.

OpenCV version 2.4.3 or higher  - Open Computer Vision framework
QT4    version 4.8.3 or higher  - QT4 framework
ccmake version 2.8.10 or higher - Curses interface for cmake
cmake  version 2.8.10 or higher - Cross-Platform Makefile Generator
gcc    version 4.7.2_2 or higher - GNU C Compiler
g++    version 4.7.2_2 or higher - GNU C++ Compiler
gcov   version 4.7.2_2 or higher - GNU coverage testing tool
lcov   version 1.10 or higher    - Graphical front-end for GCOV

BUILDING THE APPLICATION AND TEST SUITE
=======================================
1) Set the following environment variables. The exact command is dependant on the OS and shell. On a Mac OS X platform using a bash shell, this would be

> export CC = /opt/local/bin/gcc
> export CXX = /opt/local/bin/g++

2) make and cd to a build directory. The build directory should be created from
   the directory where the app was installed. So if the application was 
   located in /home/hogcv, the commands from the /home/hogcv directory would 
   be

> mkdir build
> cd build

3) Configure the build environment. Run ccmake from the build directory. If 
   all the required packages are installed, from the build directory, type
   the following command

> ccmake ..

From the curses interface, do the following:

> Press c twice to configure the read the configuration options and display
> the build options.

> Press g to generate the build files and exit the tool

4) Build the application and test suite. The build environment is setup to
   build the application and the test suite. From the build directory, run
   the following command

> make

RUNNING THE APPLICATION
=======================
After compiling, the application is located in the build/src directory
under the name hogcv. To start the application, simply type hogcv and the
main window will appear.
See the document "RunningTheApplication.pdf" for an example of how to use
the application.

RUNNING THE TEST SUITE
======================

To run the test suite, from the build directory, type the 
following:

> make test

Each test and test case will be executed and the resulted will be displayed 
on the terminal.

To display the coverage statistics, run make with the "coverage" target as
follows:

> make coverage

The build system will run the test suite, and use gcov/lcov to generate 
test results in html and open a browser window to display the results.

GENERATING DOCUMENTS
====================
The header and test source files have been commented such that doxygen can be used to generate documentation. The configuration file is located in the applications root directory under the name Doxyfile.

DIRECTORIES
===========
./CMakeLists.txt
    The CMake configuration file.
./Dalal-cvpr05.pdf
    Dalal and Triggs paper inspiring the little project.
./hdrs/
    Application header files 
./src/
    Application source files
./ui/
    QT files created using QT Designer. 
./test/
    Google test files

TO-DOS
======
The project is far from complete. The following are items that can be/should be 
done to improve the code:

1) Increase test coverage. Analysis shows that line and function coverage 
hovers around 80% and branch coverage hovers around 60%. Branch coverage should
be expanded. Conditional coverage would be next as well as dynamic profiling
to look for memory leaks.

2) Implement Set Parameters functionality. 

3) Serialize parameters. The SVM model has hardcoded parameters. A menu option 
has been added and a parameter dialog box will be created to allow the user to
change SVM parameters. We should also add functionality to load and save 
the parameters.


#include <QApplication>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "MainWin.hpp"

using namespace cv;
using namespace std;

int 
main(int argc, char** argv)
{
  QApplication app(argc,argv);
  MainWin theWin;
  theWin.show();
  
  return app.exec();
}

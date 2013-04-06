#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <QFileDialog>
#include <QDir>
#include <QString>
#include <QMessageBox>
#include "HOGCVController.hpp"
#include "MainWin.hpp"

MainWin::MainWin() : QMainWindow()
{
  setupUi(this);
  connect(actionExit,SIGNAL(triggered()),qApp,SLOT(quit()));
  connect(setPosTrainToolButton,SIGNAL(released()),this,SLOT(setPositiveTrainDir()));
  connect(setNegTrainToolButton,SIGNAL(released()),this,SLOT(setNegativeTrainDir()));
  connect(setPosTestToolButton,SIGNAL(released()),this,SLOT(setPositiveTestDir()));
  connect(setNegTestToolButton,SIGNAL(released()),this,SLOT(setNegativeTestDir()));
  connect(displayImageButton,SIGNAL(released()),this,SLOT(displayImage()));
  connect(trainButton,SIGNAL(released()),this,SLOT(train()));
  connect(classifyButton,SIGNAL(released()),this,SLOT(classify()));
}

void MainWin::setPositiveTrainDir()
{
  QFileDialog fileDialog(this);

  fileDialog.setFileMode(QFileDialog::Directory);
  if (fileDialog.exec())
  {
    QDir dir = fileDialog.directory();
    posTrainDirLabel->setText(fileDialog.directory().path());
    QStringList nameFilters;
    nameFilters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
    QStringList fileList = dir.entryList(nameFilters);
    posTrainList->clear();
    posTrainList->addItems(fileList);
  }
}

void MainWin::setNegativeTrainDir()
{
  QFileDialog fileDialog(this);

  fileDialog.setFileMode(QFileDialog::Directory);
  if (fileDialog.exec())
  {
    QDir dir = fileDialog.directory();
    negTrainDirLabel->setText(fileDialog.directory().path());
    QStringList nameFilters;
    nameFilters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
    QStringList fileList = dir.entryList(nameFilters);
    negTrainList->clear();
    negTrainList->addItems(fileList);
  }
}

void MainWin::setPositiveTestDir()
{
  QFileDialog fileDialog(this);

  fileDialog.setFileMode(QFileDialog::Directory);
  if (fileDialog.exec())
  {
    QDir dir = fileDialog.directory();
    posTestDirLabel->setText(fileDialog.directory().path());
    QStringList nameFilters;
    nameFilters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
    QStringList fileList = dir.entryList(nameFilters);
    posTestList->clear();
    posTestList->addItems(fileList);
  }
}

void MainWin::setNegativeTestDir()
{
  QFileDialog fileDialog(this);

  fileDialog.setFileMode(QFileDialog::Directory);
  if (fileDialog.exec())
  {
    QDir dir = fileDialog.directory();
    negTestDirLabel->setText(fileDialog.directory().path());
    QStringList nameFilters;
    nameFilters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
    QStringList fileList = dir.entryList(nameFilters);
    negTestList->clear();
    negTestList->addItems(fileList);
  }
}

void MainWin::displayImage()
{
  // find the selected file
  int currentTabIndex = filesTabWidget->currentIndex();
  QList<QListWidgetItem*> listItems;
  QString                 dir;

  switch(currentTabIndex)
  {
    case 0:
      dir = posTrainDirLabel->text();
      listItems = posTrainList->selectedItems();
      break;
    case 1:
      dir = negTrainDirLabel->text();
      listItems = negTrainList->selectedItems();
      break;
    case 2:
      dir = posTestDirLabel->text();
      listItems = posTestList->selectedItems();
      break;
    case 3:
      dir = negTestDirLabel->text();
      listItems = negTestList->selectedItems();
      break;
  }

    if (!listItems.empty())
    {
      QListWidgetItem* item = listItems.first();
      QString itemStr = item->text();
      std::string imgString;
      imgString += dir.toStdString();
      imgString += "/";
      imgString += itemStr.toStdString();
      HOGCVController* controller = HOGCVController::instance();

      try
      {
        controller->display(imgString);
      }
      catch(std::invalid_argument& exc)
      {
        std::cout << exc.what() << std::endl;
      }
    }
}

void MainWin::train()
{
  // 
  QList<QListWidgetItem*> listItems;
  QString                 dir;
  std::vector<std::string> fileNames;
  std::vector<int> labels;

  dir = posTrainDirLabel->text();
  if ((!dir.isNull()) && (!dir.isEmpty()))
  {
    // find the list items and add them to a vector which will be
    // passed to the train method
    for (int row=0;row < posTrainList->count();row++)
    {
      QListWidgetItem* item = posTrainList->item(row);

      std::string fileName(dir.toStdString());
      fileName += "/";
      fileName += (item->text()).toStdString();
      fileNames.push_back(fileName);
      labels.push_back(1);
    } 
  }

  dir = negTrainDirLabel->text();
  if ((!dir.isNull()) && (!dir.isEmpty()))
  {
    // find the list items and add them to a vector which will be
    // passed to the train method
    for (int row=0;row < negTrainList->count();row++)
    {
      QListWidgetItem* item = negTrainList->item(row);

      std::string fileName(dir.toStdString());
      fileName += "/";
      fileName += (item->text()).toStdString();
      fileNames.push_back(fileName);
      labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);
    } 
  }

  try
  {
    HOGCVController::instance()->train(fileNames,labels);
    QMessageBox::information(this,"HOGCv","Model trained successfully",QMessageBox::Ok);
  }
  catch(std::invalid_argument& exc)
  {
    QMessageBox::warning(this,"Error training model",exc.what(),QMessageBox::Ok);

    std::cout << exc.what() << std::endl;
  }
  catch(std::logic_error& logic_err)
  {
    QMessageBox::warning(this,"Error training model",logic_err.what(),QMessageBox::Ok);

    std::cout << logic_err.what() << std::endl;
  }
}

void MainWin::classify()
{
  // find the selected file
  int currentTabIndex = filesTabWidget->currentIndex();
  QList<QListWidgetItem*> listItems;
  QString                 dir;
  std::vector<int> labels;
  std::vector<int> predictedLabels;
  float percentageCorrect;
  int label;

  switch(currentTabIndex)
  {
    case 0:
      dir = posTrainDirLabel->text();
      listItems = posTrainList->selectedItems();
      label = 1;
      break;
    case 1:
      dir = negTrainDirLabel->text();
      listItems = negTrainList->selectedItems();
      label = -1;
      break;
    case 2:
      dir = posTestDirLabel->text();
      listItems = posTestList->selectedItems();
      label = 1;
      break;
    case 3:
      dir = negTestDirLabel->text();
      listItems = negTestList->selectedItems();
      label = -1;
      break;
  }

    if (!listItems.empty())
    {
      std::vector<std::string> fileNames;
      QList<QListWidgetItem*>::iterator iter;
      for (iter = listItems.begin();iter != listItems.end();iter++)
      {
        QListWidgetItem* item = (*iter);
        QString itemStr = item->text();
        std::string imgString;
        imgString += dir.toStdString();
        imgString += "/";
        imgString += itemStr.toStdString();
        fileNames.push_back(imgString);
        labels.push_back(label);
      }

      try
      {
        HOGCVController::instance()->classify(fileNames,labels,percentageCorrect,
                                              predictedLabels);
        for (unsigned int i=0;i < labels.size();i++)
        {
            std::cout << "Predicted lbl = " << predictedLabels[i] 
                      << "; Actual lbl = "
                      << labels[i] << std::endl;
        }
        std::cout << "Accuracy is " << std::setprecision(5) 
                  << percentageCorrect << " percent" << std::endl;

        std::string msg;
        std::ostringstream percentCorrectStr;

        percentCorrectStr << percentageCorrect;

        msg += "Classification complete.\n";
        msg += "Accuracy is ";
        msg += percentCorrectStr.str();
        msg += " % correct";

        QMessageBox::information(this,"HOGCv",msg.c_str(), QMessageBox::Ok);
      }
      catch(std::logic_error& e)
      {
        QMessageBox::warning(this,"Error classifying images",e.what(),QMessageBox::Ok);
        std::cout << e.what() << std::endl;
      }
    }
}

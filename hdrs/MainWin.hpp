#ifndef MAINWIN_H
#define MAINWIN_H

#include <QMainWindow>
#include <ui_MainWin.h>

/// @file
/// @brief Main User Interface for the HOGCv application

/// The Main Window for the HOGCV application
class MainWin : public QMainWindow, public Ui::MainWindow
{
  Q_OBJECT

public:
  /// Default constructor
  MainWin();

private slots:

  /// @brief Displays a selected image
  /// 
  /// Responds to a released() signal from the displayImageButton button.
  /// The method expects an image to be selected.
  void displayImage();

  /// @brief Trains a model based on a set of image files located in the 
  /// selected positive and negative training folders
  ///
  /// Responds to a released() signal from the trainButton button.
  void train();

  /// @brief Classify a set of selected image files based on a trained model.
  /// 
  /// Responds to a released() signal from the classifyButton button.
  /// The application expects a set of images to be selected from the positive
  /// or negative testing tabs.
  void classify();

  /// @brief Sets the directory that contains the images containing persons 
  /// that will be used for training the SVM model
  ///
  /// Responds to a released() signal from the setPosTestToolButton button.
  void setPositiveTrainDir();

  /// @brief Sets the directory that contains the images that do not contain 
  /// persons. The images will be used for training the SVM model.
  ///
  /// Responds to a released() signal from the setNegTestToolButton button.
  void setNegativeTrainDir();

  /// @brief Sets the directory that contains the images containing persons 
  /// that will be used for testing the SVM model.
  /// 
  /// Responds to a released() signal from the setPosTestToolButton button.
  void setPositiveTestDir();

  /// @brief Sets the directory that contains the images that do not contain 
  /// persons. The images will be used for testing the SVM model.
  ///
  /// Responds to a released() signal from the setNegTestToolButton button.
  void setNegativeTestDir();
};
#endif

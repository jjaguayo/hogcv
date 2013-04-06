#ifndef HOGCV_CONTROLLER_H
#define HOGCV_CONTROLLER_H

#include <string>
#include "objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "svm.h"

/// @file
/// @brief Interface for the applications main controlling class 

/// @brief Main controlling class for the HOGCv application. 
///
/// The controller is a singleton class and is used to train an SVM
/// model and use a trained SVM model to classify a set of images.
///
/// The class utilizes the libsvm library to create a support vector 
/// machine used to classify a set of images into one of two classes:
/// a class with persons in an image and a class without persons in 
/// an image.
class HOGCVController
{
public:
  /// Returns the single instance of HOGCVController
  static HOGCVController* instance();  

  /// @brief Display an image.
  /// @param fileName Full or relative path to an image file
  ///
  /// The method expects a valid path to a file that is readable using the 
  /// OpenCV2 library. 
  ///
  /// The method will display the image contained in the file.
  ///
  /// @exception std::invalid_argument 
  /// Thrown if the path to the file is invalid or if the file cannot be read
  /// using the OpenCV2 library.
  void display(std::string fileName);  

  /// @brief Train an SVM given a set of images and labels.
  /// @param fileNames 
  /// Vector of full or relative paths to image files
  /// @param labels    
  /// Vector of labels indicating if a person is in the image or not.
  ///
  /// The train method is used to train an SVM model.
  /// Currently, the model uses a radial basis function kernel and the
  /// parameters used to train the model cannot be modified at runtime.
  ///
  /// The model expects a a set of file names (full or relative paths)
  /// and a set of labels corresponding to whether a person is in an
  /// image or not. The image files must be readable using the OpenCV2 
  /// library.
  ///
  /// @exception std::invalid_argument
  /// Thrown if no filenames or labels are supplied, an invalid file name or 
  /// label is given or if the number of file names and labels are not equal
  ///
  /// @exception std::logic_error
  /// Thrown if the parameters used to train the model are not valid.
  /// This exception should not be thrown until the functionality to
  /// change training and model parameters is implemented.
  void train(std::vector<std::string> fileNames, std::vector<int> labels); 

  ///  @brief Predict the labels for a set of images based on a trained SVM 
  ///  model.
  /// @param fileNames 
  ///        Vector of full or relative paths to image files
  /// @param actualLabels    
  ///        Vector of labels indicating if a person is in the image or not
  /// @param percentageCorrect 
  ///        The percentage of images whose labels were predicted correctly.
  /// @param predictedLabels 
  ///        A vector of predicted labels
  ///
  /// The classify method predicts a class for a given set of image files.
  /// The image files must be readable using the OpenCV2 library. 
  ///
  /// The number of actual labels must equal the number of filenames given.
  ///
  /// The method will calculate the percent of predictions that were correct
  /// and return the percent as well as the predicted labels for each file.
  ///
  /// @exception std::invalid_argument
  /// Thrown if no filenames or actual labels are supplied, an invalid file 
  /// name or label is given or if the number of file names and actual labels 
  /// are not equal
  ///
  /// @exception std::logic_error
  /// Thrown if the method is called but a model has not been trained.
  void classify(const std::vector<std::string>& fileNames, const std::vector<int>& actualLabels, float& percentageCorrect, std::vector<int>& predictedLabels);

  /// @brief Frees any data allocated to an SVM model
  void freeModelContent();

  /// Positive label indicating a person is in the image
  static const int PERSON_IN_IMAGE; 

  /// Negative label indicating a person is note in the image
  static const int NO_PERSON_IN_IMAGE; 

protected:
  /// Singleton instance of the controller
  static HOGCVController* inst;          

  /// The features of each data point to be used for training or classifying 
  /// as well as the label associated with each data point.
  struct svm_problem problem;            

  /// The parameters used to train the model
  struct svm_parameter parameters;       

  /// The SVM model
  struct svm_model *model;               

  /// Default constructor
  HOGCVController();

  /// Destructor
  ~HOGCVController();

private:

  /// Free memory allocated for the svm_problem struct
  void cleanUpSvmModel();

  /// @brief Retrieve descriptors for an image
  /// @param image 
  ///        A matrix of pixel values for an image
  /// @param allDescriptorValues 
  ///        The descriptors related to the image
  /// @param blockSizeX 
  ///        The number of blocks to break the image into in the horizontal 
  ///        direction
  /// @param blockSizeY 
  ///        The number of blocks to break the image into in the vertical 
  ///        direction
  /// @param cellSizeX 
  ///        The number of cells to break each block into in the horizontal 
  ///        direction
  /// @param cellSizeY The number of cells to break each block into in the 
  ///        vertical direction
  void retrieveDescriptors(cv::Mat image,std::vector<float> &allDescriptorValues,int blockSizeX, int blockSizeY, int cellSizeX, int cellSizeY);

  /// @brief Retrieve descriptors from a block in an image
  /// @param gradMag 
  ///        A matrix of pixel gradients magnitudes for an image 
  /// @param gradOrient 
  ///        A matrix of pixel gradient orientations for an image 
  /// @param allCellDescriptorValues 
  ///        Descriptors for each cell in the block
  /// @param startX 
  ///        The horizontal pixel location in the gradient and orientation 
  ///        matrices to start using when determining the descriptors
  /// @param startY 
  ///        The vertical pixel location in the gradient and orientation 
  ///        matrices to start using when determining the descriptors
  /// @param blockSizeX 
  ///        The number of blocks to break the image into in the horizontal 
  ///        direction
  /// @param blockSizeY 
  ///        The number of blocks to break the image into in the vertical 
  ///        direction
  /// @param cellSizeX 
  ///        The number of cells to break each block into in the horizontal 
  ///        direction
  /// @param cellSizeY 
  ///        The number of cells to break each block into in the vertical 
  ///        direction
  void retrieveDescriptorsFromBlock(cv::Mat gradMag, cv::Mat gradOrient,std::vector<float> &allCellDescriptorValues,int startX, int startY, int blockSizeX, int blockSizeY, int cellSizeX, int cellSizeY);

  /// @brief Retrieve descriptors from a cell
  /// @param gradMag 
  ///        A matrix of pixel gradient magnitudes for an image
  /// @param gradOrient 
  ///        A matrix of pixel gradient orientations for an image
  /// @param cellDescriptorValues 
  ///        Descriptors for a given cell
  /// @param startX 
  ///        The horizontal pixel location in the gradient and orientation matrices to start using when determining the descriptors
  /// @param startY 
  ///        The vertical pixel location in the gradient and orientation 
  ///        matrices to start using when determining the descriptors
  /// @param cellSizeX 
  ///        The number of cells to break each block into in the horizontal 
  ///        direction
  /// @param cellSizeY 
  ///        The number of cells to break each block into in the vertical 
  ///        direction
  void retrieveDescriptorsFromCell(cv::Mat gradMag, cv::Mat gradOrient, std::vector<float> &cellDescriptorValues, int startX, int startY, int cellSizeX, int cellSizeY);

  /// @brief Given two matrices of the gradient magnitudes in the x and y directions for an image, determine the overall gradient magnitudes.
  /// @param gradX 
  ///        A matrix of gradient magnitudes in the x direction 
  /// @param gradY 
  ///        A matrix of gradient magnitudes in the y direction 
  /// @param gradMagnitude 
  ///        A matrix of the overall gradient magnitudes 
  void calculateGradMagnitude(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& gradMagnitude);

  /// @brief Given two matrices of the gradient orientations in the x and y directions for an image, determine the overall gradient orientations.
  /// @param gradX 
  ///        A matrix of gradient orientations in the x direction 
  /// @param gradY 
  ///        A matrix of gradient orientations in the y direction 
  /// @param gradMagnitude 
  ///        A matrix of the overall gradient magnitudes 
  void calculateGradOrientation(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& gradOrientation);

  /// @brief Determine the gradient magnitudes and orientations for an image
  /// @param image 
  ///        A matrix of pixel values for an image
  /// @param gradMag 
  ///        Gradient magnitudes for an image
  /// @param gradOrient 
  ///        Gradient orientations for an image
  void calculateGradients(const cv::Mat& image, cv::Mat& gradMag, cv::Mat& gradOrient);
};
#endif


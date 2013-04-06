#include <iostream>
#include <HOGCVController.hpp>
#include <svm.h>
#include <stdexcept>

#include <gtest/gtest.h>

/// @file
/// @brief Tests for the HOGCVController class
namespace TestHOGCV
{
  /// @brief Google test fixture class to test functionality of the HOGCVController class.
  class HOGCVControllerTest : public ::testing::Test
  {
    protected:
    /// Singleton instance of the HOGCVController class
    HOGCVController* controllerInst;        
    /// Image to load for a given test
    std::string person_bike_bmp;            

    /// Function used to suppress superfluous output from svm library
    static void localPrintFunc(const char * msg) { }

    /// @brief Sets the location of the image file and ensure the 
    /// HOGCVController is instantiated before running the tests
    virtual void SetUp()
    {
      person_bike_bmp = std::string("person_and_bike_001.bmp");
      controllerInst = HOGCVController::instance();
    }
  };

/// @brief Test the display method with an invalid filename
TEST_F(HOGCVControllerTest, testDisplayFunctionWithInvalidFileName)
{
  EXPECT_THROW(controllerInst->display(""),std::invalid_argument);
}

/// @brief Test the display method with an valid filename
TEST_F(HOGCVControllerTest, testDisplayFunctionWithValidFileName)
{
  EXPECT_NO_THROW(controllerInst->display(person_bike_bmp.c_str()));
}

/// @brief Test the train method with an invalid filename
TEST_F(HOGCVControllerTest, testTrainFunctionWithInvalidFileName)
{
  std::vector<std::string> fileNames;
  std::vector<int> labels;

  fileNames.push_back("");
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  EXPECT_THROW(controllerInst->train(fileNames,labels),std::invalid_argument);
}

/// @brief Test the train method with an valid filename
TEST_F(HOGCVControllerTest, testTrainFunctionWithValidFileName)
{
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  svm_set_print_string_function(localPrintFunc);

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);
  EXPECT_NO_THROW(controllerInst->train(fileNames,labels));
}

/// @brief Test the train method with an invalid label
TEST_F(HOGCVControllerTest, testTrainFunctionWithInvalidLabel)
{
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(0);
  EXPECT_THROW(controllerInst->train(fileNames,labels),std::invalid_argument);
}

/// @brief Test the train method with different sized parameters 
TEST_F(HOGCVControllerTest, testTrainFunctionWithDifferentSizedParameters)
{
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  EXPECT_THROW(controllerInst->train(fileNames,labels),std::invalid_argument);
}

/// Test the train method with no files or labels
TEST_F(HOGCVControllerTest, testTrainFunctionWithNoFileNamesOrLabels)
{
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  EXPECT_THROW(controllerInst->train(fileNames, labels),std::invalid_argument);
}

/// Test the classify method with invalid filename 
TEST_F(HOGCVControllerTest, testClassifyFunctionWithInvalidFileName)
{
  std::vector<std::string> fileNames;     
  std::vector<int> labels;
  float percentageCorrect;
  std::vector<int> predictedLabels;

  fileNames.push_back("");
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  EXPECT_THROW(controllerInst->classify(fileNames,labels,percentageCorrect,predictedLabels),std::invalid_argument);
}

/// Test the classify method with no trained model 
TEST_F(HOGCVControllerTest, testClassifyFunctionWithNoTrainedModel)
{
  float percentageCorrect;
  std::vector<int> predictedLabels;
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  controllerInst->freeModelContent();

  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);

  EXPECT_THROW(controllerInst->classify(fileNames,labels,percentageCorrect,predictedLabels),std::logic_error);
}

/// Test the classify method with valid filename
TEST_F(HOGCVControllerTest, testClassifyFunctionWithValidFileName)
{
  float percentageCorrect;
  std::vector<int> predictedLabels;
  std::vector<std::string> fileNames;     
  std::vector<int> labels;

  svm_set_print_string_function(localPrintFunc);

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);
  controllerInst->train(fileNames, labels);
  EXPECT_NO_THROW(controllerInst->classify(fileNames,labels,percentageCorrect,predictedLabels));
}

/// @brief Test the classify method with no filenames or labels
TEST_F(HOGCVControllerTest, testClassifyFunctionWithNoFileNamesOrLabels)
{
  float percentageCorrect;
  std::vector<int> predictedLabels;
  std::vector<std::string> fileNames;     
  std::vector<std::string> fileNames2;
  std::vector<int> labels;
  std::vector<int> labels2;

  svm_set_print_string_function(localPrintFunc);

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);

  controllerInst->train(fileNames, labels);

  EXPECT_THROW(controllerInst->classify(fileNames2,labels2,percentageCorrect,predictedLabels),std::logic_error);
}

/// @brief Test the classify method with different sized parameters
TEST_F(HOGCVControllerTest, testClassifyFunctionWithDifferentSizeParams)
{
  float percentageCorrect;
  std::vector<int> predictedLabels;
  std::vector<std::string> fileNames;     
  std::vector<std::string> fileNames2;     
  std::vector<int> labels;
  std::vector<int> labels2;

  svm_set_print_string_function(localPrintFunc);

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);

  fileNames2.push_back(person_bike_bmp.c_str());

  controllerInst->train(fileNames, labels);
  EXPECT_THROW(controllerInst->classify(fileNames2,labels2,percentageCorrect,predictedLabels),std::invalid_argument);
}

/// @brief Test the classify method with an invalid label
TEST_F(HOGCVControllerTest, testClassifyFunctionWithInvalidLabel)
{
  float percentageCorrect;
  std::vector<int> predictedLabels;
  std::vector<std::string> fileNames;     
  std::vector<int> labels;
  std::vector<int> labels2;

  svm_set_print_string_function(localPrintFunc);

  fileNames.push_back(person_bike_bmp.c_str());
  fileNames.push_back(person_bike_bmp.c_str());
  labels.push_back(HOGCVController::PERSON_IN_IMAGE);
  labels.push_back(HOGCVController::NO_PERSON_IN_IMAGE);

  controllerInst->train(fileNames, labels);

  labels2.push_back(0);

  EXPECT_THROW(controllerInst->classify(fileNames,labels2,percentageCorrect,predictedLabels),std::invalid_argument);
}
}

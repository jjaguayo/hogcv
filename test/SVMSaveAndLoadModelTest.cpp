#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Test the functions used to save and load a generated model
namespace SVMLibraryTests
{
  /// @brief Google test fixture class to test functionality of the svm library
  /// related to saving and loading generated model data
  class SVMSaveAndLoadModelTest: public ::testing::Test
  {
    protected:
    /// Number of sample data points to train the model with
    int numSampleDataPoints;

    /// Number of features for each data point
    int numValuesPerDataPoint;

    /// The file to store the model to
    std::string fileName;

    /// The sample labeled data
    svm_problem*   problem;

    /// The parameters used when training the model
    svm_parameter* param;

    /// The SVM model to save
    svm_model*     model;

    /// The SVM model to load
    svm_model*     model2;

    /// @brief Setups up the data for each test
    /// Initializes the number of sample data points and number of features per
    /// data point as well as the svm_problem and svm_param structures used
    /// to train the model. It also sets the name of the file to use
    virtual void SetUp()
    {
      problem = new svm_problem;
      param = new svm_parameter;
      model = NULL;
      model2 = NULL;
      numSampleDataPoints = 6;
      numValuesPerDataPoint = 2;
      fileName = "model.tmp";

      // create problem structure
      problem->l = numSampleDataPoints;
      problem->y = new double[numSampleDataPoints];
      for (int i=0;i<problem->l;i++)
      {
        problem->y[i] = 1;
      }

      // use 6 sample data points with 2 values per data point
      problem->x = new struct svm_node*[numSampleDataPoints];

      for (int i = 0; i < numSampleDataPoints;i++)
      {
        problem->x[i] = new struct svm_node[numValuesPerDataPoint+1];
        for (int j = 0;j < numValuesPerDataPoint;j++)
        {
          problem->x[i][j].index = j+1;
          problem->x[i][j].value = (double) j+1;
        }
        problem->x[i][numValuesPerDataPoint].index = -1;
        problem->x[i][numValuesPerDataPoint].value = 0;
      }

      // create parameter structure 
      param->svm_type = ONE_CLASS; 
      param->kernel_type = RBF;
      param->gamma = 1;
      param->degree = 1;
      param->cache_size = 100;
      param->eps = 1;
      param->C = 2;
      param->nu = 0.5;
      param->p = 1;
      param->shrinking = 0;
      param->probability = 0;
      param->coef0 = 0;
      param->weight = new double;
      param->weight_label = new int;

      model = NULL;
    }

    /// @brief Free the allocated memory
    virtual void TearDown()
    {
      for (int i = 0;i < numSampleDataPoints;i++)
      {
        delete [] problem->x[i];
      }
  
      delete [] problem->x;
      delete [] problem->y;
      delete problem;

      svm_destroy_param(param);
      delete param;

      if (NULL != model)
      {
        svm_free_model_content(model);
        delete model;
        model = NULL;
      }

      if (NULL != model2)
      {
        svm_free_model_content(model2);
        delete model2;
        model = NULL;
      }
    }
  };

/// @brief Statement coverage test for the svm_save_model and svm_load_model
/// functions
TEST_F(SVMSaveAndLoadModelTest,testSaveAndLoadModelFunctions)
{
  model = svm_train(problem, param);
  ASSERT_EQ(0,svm_save_model(fileName.c_str(),model));
  svm_free_model_content(model);

  model2 = svm_load_model(fileName.c_str());
  EXPECT_TRUE (NULL != model2);
  svm_free_model_content(model2);
}
}

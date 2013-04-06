#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Tests for the svm_predict function 
namespace SVMLibraryTests
{
  /// @brief Google test fixture for testing the svm_predict function
  class SVMPredictTest: public ::testing::Test
  {
    protected:
    /// Number of sample data points to train the model with
    int numSampleDataPoints;

    /// Number of features for each data point
    int numValuesPerDataPoint;

    /// The sample labeled data
    svm_problem*   problem;

    /// The parameters used when training the model
    svm_parameter* param;

    /// The SVM model
    svm_model*     model;

    /// @brief Setups up the data for each test
    ///
    /// Initializes the number of sample data points and number of features per
    /// data point as well as the svm_problem and svm_param structures used
    /// to train the model
    virtual void SetUp()
    {
      numSampleDataPoints = 1000;
      numValuesPerDataPoint = 2;
      problem = new svm_problem;
      param = new svm_parameter;

      // create problem structure
      problem->l = numSampleDataPoints;
      problem->y = new double[numSampleDataPoints];
      for (int i=0;i<problem->l;i++)
      {
        if (i <(numSampleDataPoints / 2))
          problem->y[i] = 1;
        else
          problem->y[i] = -1;
      }

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
      param->gamma = 1/numSampleDataPoints;
      param->degree = 1;
      param->cache_size = 100;
      param->eps = 0.001;
      param->C = 2;
      param->nu = 0.5;
      param->p = 0.01;
      param->shrinking = 1;
      param->probability = 0;
      param->coef0 = 0;
      param->weight = new double[2];
      param->weight_label = new int[2];
      param->weight[0] = -1;
      param->weight_label[0] = -1;
      param->weight[1] = 1;
      param->weight_label[1] = 1;
      param->nr_weight = 2;
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
    }
  };

/// @brief Statement coverage test for the svm_predict function
TEST_F(SVMPredictTest, testPredictFunction)
{
  double         predictedValue;

  model = svm_train(problem, param);

  predictedValue = svm_predict(model, problem->x[0]);

  EXPECT_EQ(-1, predictedValue);
}
}

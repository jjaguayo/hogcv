#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Tests for the svm_cross_validation function
namespace SVMLibraryTests
{
  /// @brief Google test fixture for testing the svm_cross_validation function
  class SVMCrossValidationTest: public ::testing::Test
  {
    protected:
    /// Number of sample data points to train the model with
    int numSampleDataPoints;

    /// Number of features for each data point
    int numValuesPerDataPoint;

    /// Number of ways to fold the sample data
    int numFolds;

    /// The sample labeled data
    svm_problem*   problem;

    /// The parameters used when training the model
    svm_parameter* param;

    /// @brief Setups up the data for each test
    /// Initializes the number of sample data points and number of features per
    /// data point as well as the svm_problem and svm_param structures used
    /// to train the model
    virtual void SetUp()
    {
      numSampleDataPoints = 100;
      numValuesPerDataPoint = 2;
      numFolds = 5;
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
      param->svm_type = C_SVC; 
      param->kernel_type = RBF;
      param->gamma = 1;
      param->degree = 1;
      param->cache_size = 100;
      param->eps = 0.001;
      param->C = 2;
      param->nu = 0.5;
      param->p = 0.001;
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

    /// Free up allocated data
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
    }
  };

/// @brief Statement coverage test for the svm_cross_validation function
TEST_F(SVMCrossValidationTest,testCrossValidationFunction)
{
  double* targets = new double[numSampleDataPoints];

  EXPECT_TRUE(NULL == svm_check_parameter(problem,param));

  EXPECT_NO_THROW(svm_cross_validation(problem, param,numFolds,targets));
}
}

#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Tests for the svm_train function 
namespace SVMLibraryTests
{
  /// @brief Google test fixture for testing the svm_train function
  class SVMTrainingTest: public ::testing::Test
  {
    protected:

    /// Number of sample data points to train the model with
    int numSampleDataPoints;

    /// Number of features for each data point
    int numValuesPerDataPoint;

    /// @brief Suppresses the output to stdout
    ///
    /// Function used to suppress superfluous output from svm library
    /// The library prints messages to stdout which are irrelevant to
    /// the test case. This function allows the test case to suppress
    /// that output.
    static void localPrintFunc(const char * msg) { }

    /// @brief Setups up the data for each test
    ///
    /// Initializes the number of sample data points and number of features per
    /// data point
    virtual void SetUp()
    {
      numSampleDataPoints = 6;
      numValuesPerDataPoint = 2;
    }

    /// @brief Initialize the sample data points and the parameters used to train
    /// the model
    void localSetup(svm_problem* localproblem, svm_parameter* localparam,
                    svm_model*   localmodel,int numLabels)
    {
      // create problem structure
      localproblem->l = numSampleDataPoints;
      localproblem->y = new double[numSampleDataPoints];
      for (int i=0;i<localproblem->l;i++)
      {
        if (2 == numLabels)
        {
          if (i < (numSampleDataPoints /2))
          {
            localproblem->y[i] = 1;
          }
          else
          {
            localproblem->y[i] = -1;
          }
        }
        else
        {
          localproblem->y[i] = 1;
        }
      }

      localproblem->x = new struct svm_node*[numSampleDataPoints];

      for (int i = 0; i < numSampleDataPoints;i++)
      {
        localproblem->x[i] = new struct svm_node[numValuesPerDataPoint+1];
        for (int j = 0;j < numValuesPerDataPoint;j++)
        {
          localproblem->x[i][j].index = j+1;
          localproblem->x[i][j].value = (double) j+1;
        }
        localproblem->x[i][numValuesPerDataPoint].index = -1;
        localproblem->x[i][numValuesPerDataPoint].value = 0;
      }

      // create parameter structure 
      localparam->svm_type = ONE_CLASS; 
      localparam->kernel_type = RBF;
      localparam->gamma = 1;
      localparam->degree = 1;
      localparam->cache_size = 100;
      localparam->eps = 1;
      localparam->C = 2;
      localparam->nu = 0.5;
      localparam->p = 1;
      localparam->shrinking = 0;
      localparam->probability = 0;
      localparam->coef0 = 0;
      localparam->weight = new double[2];
      localparam->weight_label = new int[2];
      localparam->weight[0] = -1;
      localparam->weight[1] = 1;
      localparam->weight_label[0] = -1;
      localparam->weight_label[1] = 1;
      localparam->nr_weight = 2;

      localmodel = NULL;
    }
 
    /// Free allocated memory
    void localTeardown(svm_problem* localproblem, svm_parameter* localparam,
                       svm_model* localmodel)
    {
      for (int i = 0;i < numSampleDataPoints;i++)
      {
        delete [] localproblem->x[i];
      }
  
      delete [] localproblem->x;
      delete [] localproblem->y;
      delete localproblem;

      svm_destroy_param(localparam);
      delete localparam;

      if (NULL != localmodel)
      {
        svm_free_model_content(localmodel);
        delete localmodel;
        localmodel = NULL;
      }
    }
  };

/// @brief Train a NU_SVR model with an RBF kernel
TEST_F(SVMTrainingTest, testSvmTrainWithNuSvrRbfKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  // Train a model with one label (i.e. regression model)
  localSetup(localproblem, localparam, localmodel,1);

  localparam->svm_type = NU_SVR; 
  localparam->kernel_type = RBF;
  // use to suppress superfluous output
  svm_set_print_string_function(localPrintFunc);

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  localTeardown(localproblem, localparam, localmodel);
}

/// @brief Train a NU_SVC model with an RBF kernel
TEST_F(SVMTrainingTest, testSvmTrainWithNuSvcRbfKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  svm_set_print_string_function(localPrintFunc);

  // Train a model with two labels (i.e. a classification model)
  localSetup(localproblem, localparam, localmodel,2);

  localparam->svm_type = NU_SVC; 
  localparam->kernel_type = RBF;

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  localTeardown(localproblem, localparam, localmodel);
}

/// @brief Train a EPSILON_SVR model with an RBF kernel
TEST_F(SVMTrainingTest, testSvmTrainWithEpsilonSvrRbfKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  localSetup(localproblem, localparam, localmodel,1);

  localparam->svm_type = EPSILON_SVR; 
  localparam->kernel_type = RBF;
  // use to suppress superfluous output
  svm_set_print_string_function(localPrintFunc);

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  localTeardown(localproblem, localparam, localmodel);
}


/// @brief Train a ONE_CLASS model with an RBF kernel
TEST_F(SVMTrainingTest, testSvmTrainWithOneClassRbfKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  localSetup(localproblem, localparam, localmodel,2);

  localparam->svm_type = ONE_CLASS; 
  localparam->kernel_type = RBF;

  svm_set_print_string_function(localPrintFunc);

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  localTeardown(localproblem, localparam, localmodel);
}

/// @brief Train a ONE_CLASS model with an Linear kernel
TEST_F(SVMTrainingTest, testSvmTrainWithOneClassLinearKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  localSetup(localproblem, localparam, localmodel,2);

  localparam->svm_type = ONE_CLASS; 
  localparam->kernel_type = LINEAR;
  // use to suppress superfluous output
  svm_set_print_string_function(localPrintFunc);

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  // clean up
  localTeardown(localproblem,localparam,localmodel);
}

/// @brief Train a ONE_CLASS model with an Polynomial kernel
TEST_F(SVMTrainingTest, testSvmTrainWithOneClassPolynomialKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  localSetup(localproblem, localparam, localmodel,2);

  localparam->svm_type = ONE_CLASS; 
  localparam->kernel_type = POLY;
  // use to suppress superfluous output
  svm_set_print_string_function(localPrintFunc);

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  // clean up
  localTeardown(localproblem,localparam,localmodel);
}

/// @brief Train a ONE_CLASS model with an sigmoid kernel
TEST_F(SVMTrainingTest, testSvmTrainWithOneClassSigmoidKernel)
{
  svm_problem*   localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;
  svm_model*     localmodel;
  localmodel = NULL;

  localSetup(localproblem, localparam, localmodel,2);

  // use to suppress superfluous output
  svm_set_print_string_function(localPrintFunc);

  localparam->svm_type = ONE_CLASS; 
  localparam->kernel_type = SIGMOID;

  // train model
  localmodel = svm_train(localproblem, localparam);
  
  // verify model is not null and values are correct based on input
  EXPECT_TRUE(NULL != localmodel);
  EXPECT_EQ(2,svm_get_nr_class(localmodel));

  // clean up
  localTeardown(localproblem,localparam,localmodel);
}
}

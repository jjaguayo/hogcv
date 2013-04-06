#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Tests for the svm_check_parameter function
namespace SVMLibraryTests
{
  /// @brief Google test fixture used for testing the svm_check_parameter function
  class SVMCheckParamsTest: public ::testing::Test
  {
    protected:

    /// The sample labeled data
    struct svm_problem *problem;

    /// Number of sample data points to train the model with
    int numSampleDataPoints;

    /// Number of features for each data point
    int numValuesPerDataPoint;

    /// @brief Setups up the data for each test
    ///
    /// Initializes the number of sample data points and number of features per
    /// data point as well as the svm_problem and svm_param structures used
    /// to train the model
    virtual void SetUp()
    {
      numSampleDataPoints = 6;
      numValuesPerDataPoint = 2;
      problem = new svm_problem;
      // create problem structure
      problem->l = numSampleDataPoints;
      problem->y = new double[numSampleDataPoints];
      for (int i=0;i < problem->l;i++)
      {
        problem->y[i] = 1;
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
    }
  };

/// @brief Test the svm_check_parameters function with known and unknown svm types
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithKnownAndUnknownSVMTypes)
{
  std::string unknownTypeMsg = "unknown svm type";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = NU_SVC;
  localproblem->l = 0;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = ONE_CLASS;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = EPSILON_SVR;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = NU_SVR;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = -1;
  EXPECT_STREQ(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with known and unknown kernel types
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithKnownAndUnknownKernelTypes)
{
  std::string unknownTypeMsg = "unknown kernel type";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  localparam->kernel_type = LINEAR;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->kernel_type = POLY;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->kernel_type = RBF;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->kernel_type = SIGMOID;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->kernel_type = PRECOMPUTED;
  EXPECT_STRNE(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->kernel_type = -1;
  EXPECT_STREQ(unknownTypeMsg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the gamma parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryGammaValues)
{
  std::string gammaMsg = "gamma < 0";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  EXPECT_STRNE(gammaMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->gamma = -1;
  EXPECT_STREQ(gammaMsg.c_str(),svm_check_parameter(localproblem,localparam));
  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the degree parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryDegreeValues)
{
  std::string degreeMsg = "degree of polynomial kernel < 0";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  EXPECT_STRNE(degreeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->degree = -1;
  EXPECT_STREQ(degreeMsg.c_str(),svm_check_parameter(localproblem,localparam));
  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the cache_size parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryCacheSizeValues)
{
  std::string cacheMsg = "cache_size <= 0";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  EXPECT_STRNE(cacheMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->cache_size = 0;
  EXPECT_STREQ(cacheMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->cache_size = -1;
  EXPECT_STREQ(cacheMsg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the eps parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryEpsValues)
{
  std::string epsMsg = "eps <= 0";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->svm_type = C_SVC;
  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  EXPECT_STRNE(epsMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->eps = 0;
  EXPECT_STREQ(epsMsg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->eps = -1;
  EXPECT_STREQ(epsMsg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the nu parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryNuValues)
{
  std::string msg = "nu <= 0 or nu > 1";
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->svm_type = C_SVC;
  localparam->nu = 0.5;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = -1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->svm_type = NU_SVC;
  localparam->nu = 2;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 0;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = -1;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->svm_type = ONE_CLASS;
  localparam->nu = 2;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 0;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = -1;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->svm_type = NU_SVR;
  localparam->nu = 2;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = 0;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));
  localparam->nu = -1;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(problem,localparam));

  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the p parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryPValues)
{
  std::string msg = "p < 0";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->nu = 2;
  localparam->svm_type = C_SVC;
  localparam->p = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->p = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->p = -1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = EPSILON_SVR;
  localparam->p = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->p = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->p = -1;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the shrinking parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryShrinkingValues)
{
  std::string msg = "shrinking != 0 and shrinking != 1";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->nu = 2;
  localparam->svm_type = C_SVC;
  localparam->p = 1;

  localparam->shrinking = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->shrinking = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->shrinking = 2;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with boundary values for the probability parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithBoundaryProbValues)
{
  std::string msg = "probability != 0 and probability != 1";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->nu = 2;
  localparam->svm_type = C_SVC;
  localparam->p = 1;
  localparam->shrinking = 0;
  localparam->probability = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->probability = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->probability = 2;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Test the svm_check_parameters function with svm type set to ONE_CLASS and boundary values for the probability parameter
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionWithOneClassSVMTypeAndBoundaryProbValues)
{
  std::string msg = "one-class SVM probability output not supported yet";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->nu = 1;
  localparam->p = 1;
  localparam->shrinking = 0;
  localparam->svm_type = C_SVC;
  localparam->probability = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = C_SVC;
  localparam->probability = 1;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = ONE_CLASS;
  localparam->probability = 0;
  EXPECT_STRNE(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localparam->svm_type = ONE_CLASS;
  localparam->probability = 1;
  EXPECT_STREQ(msg.c_str(),svm_check_parameter(localproblem,localparam));

  delete localproblem;
  delete localparam;
}

/// @brief Statement coverage test for the svm_check_parameters function. 
/// The test executes the code checking the feasability of building a model with an svm type of NU_SVC and a set of valid model parameters.
TEST_F(SVMCheckParamsTest,testCheckParameterFunctionNuSVCClassificationFeasability)
{
  std::string msg = "specified nu is infeasible";
  svm_problem* localproblem = new svm_problem;
  svm_parameter* localparam = new svm_parameter;

  localparam->kernel_type = LINEAR;
  localparam->gamma = 1;
  localparam->degree = 1;
  localparam->cache_size = 1;
  localparam->eps = 1;
  localparam->C = 2;
  localparam->nu = 1;
  localparam->p = 1;
  localparam->shrinking = 0;
  localparam->svm_type = NU_SVC;
  localparam->probability = 0;

  localproblem->l = 6;
  localproblem->y = new double[6];
  localproblem->y[0] = 1;
  localproblem->y[1] = 1;
  localproblem->y[2] = 1;
  localproblem->y[3] = 1;
  localproblem->y[4] = -1;
  localproblem->y[5] = -1;

  EXPECT_STREQ(msg.c_str(),svm_check_parameter(localproblem,localparam));
  localproblem->y[3] = -1;
  EXPECT_STREQ(NULL,svm_check_parameter(localproblem,localparam));
  localproblem->y[3] = 1;
  localparam->nu = 0.5;
  EXPECT_STREQ(NULL,svm_check_parameter(localproblem,localparam));

  delete [] localproblem->y;
  delete localproblem;
  delete localparam;
}
}

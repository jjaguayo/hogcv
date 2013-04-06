#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Test the get functions in the svm library 
namespace SVMLibraryTests
{

  /// @brief Google test fixture class to test functionality of the svm library 
  /// related to retrieving model and parameter information
  class SVMGetFunctionsTest: public ::testing::Test
  {
    protected:
    /// SVM Model used by each test
    struct svm_model *model;

    /// Initializes the svm_model before running the tests
    virtual void SetUp()
    {
      model = new svm_model;
      model->param.svm_type = C_SVC;
      model->nr_class       = 2;
      model->label = new int[2];
      model->label[0] = 0;
      model->label[1] = 1;
      model->l     = 2;
      model->sv_indices = new int[2];
      model->sv_indices[0] = 0;
      model->sv_indices[1] = 1;
    }

    /// Frees up data allocated for the model
    virtual void TearDown()
    {
      delete [] model->label;
      delete [] model->sv_indices;
      delete model;
    }
  };
  
/// @brief Test the svm_get_svm_type function
///
TEST_F(SVMGetFunctionsTest, testGetSvmTypeFunction)
{
  EXPECT_EQ(C_SVC,svm_get_svm_type(model));
}

/// @brief Test the svm_get_nr_class function
///
TEST_F(SVMGetFunctionsTest, testGetNrClassFunction)
{
  EXPECT_EQ(2,svm_get_nr_class(model));
}

/// @brief Test the svm_get_model_labels function
///
TEST_F(SVMGetFunctionsTest, testGetModelLabelsFunction)
{
  int *labels;
  labels = new int[2];
  svm_get_labels(model,labels);
  EXPECT_EQ(0,labels[0]);
  EXPECT_EQ(1,labels[1]);
  delete [] labels;
}

/// @brief Test the svm_get_sv_indices function
///
TEST_F(SVMGetFunctionsTest, testGetSvIndicesFunction)
{
  int *indices;
  indices = new int[2];
  svm_get_sv_indices(model,indices);
  EXPECT_EQ(0,indices[0]);
  EXPECT_EQ(1,indices[1]);
  delete [] indices;
}

/// @brief Test the svm_get_nr_sv function
///
TEST_F(SVMGetFunctionsTest, testGetNrSvFunction)
{
  EXPECT_EQ(2,svm_get_nr_sv(model));
}

/// @brief Test the svm_get_svr_probability function
TEST_F(SVMGetFunctionsTest, testGetSvrProbabilityFunction)
{
  model->param.svm_type = C_SVC;
  model->probA = NULL;
  EXPECT_EQ(0,svm_get_svr_probability(model)); 
  model->probA = new double(0);
  model->param.svm_type = EPSILON_SVR;
  EXPECT_EQ(0,svm_get_svr_probability(model)); 
  model->param.svm_type = NU_SVR;
  EXPECT_EQ(0,svm_get_svr_probability(model)); 
  model->probA[0] = 10;
  model->param.svm_type = EPSILON_SVR;
  EXPECT_EQ(10,svm_get_svr_probability(model)); 
  model->param.svm_type = NU_SVR;
  EXPECT_EQ(10,svm_get_svr_probability(model)); 
  model->param.svm_type = C_SVC;
  delete model->probA;
  model->probA = NULL;
  EXPECT_EQ(0,svm_get_svr_probability(model)); 
}
}

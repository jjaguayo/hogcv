#include <iostream>
#include <svm.h>
#include <gtest/gtest.h>

/// @file
/// @brief Test the svm library functions that are used to free allocated memory
namespace SVMLibraryTests
{
  /// Google test fixture for testing the svm library functions used to free allocated memory
  class SVMFreeMemoryTest : public ::testing::Test
  {
  };

/// @brief Test the svm_destory_param function
TEST_F(SVMFreeMemoryTest, testDestroyParamFunction)
{
  svm_parameter* localparam = new svm_parameter;

  localparam->weight = new double;
  localparam->weight_label = new int;
  localparam->weight[0] = 0;
  localparam->weight_label[0] = 0;
  EXPECT_EQ(0,localparam->weight[0]);
  EXPECT_EQ(0,localparam->weight_label[0]);

  svm_destroy_param(localparam);
  EXPECT_EQ(NULL,localparam->weight);
  EXPECT_EQ(NULL,localparam->weight_label);

  delete localparam;

}
}

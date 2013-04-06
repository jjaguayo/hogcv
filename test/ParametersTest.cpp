#include <iostream>
#include <Parameters.hpp>
#include <svm.h>

#include <gtest/gtest.h>

/// @file
/// @brief Tests for the Parameters class
namespace TestHOGCV
{
  /// @brief Google test fixture class to test functionality of the Parameters class.
  class ParametersTest : public ::testing::Test
  {
    protected:
    /// Singleton instance of the Parameters class
    Parameters* paramInst;

    /// @brief Instantiates the Parameter class before running the tests
    virtual void SetUp()
    {
      paramInst = Parameters::instance();
    }
  };

/// @brief Test that the default parameter values are set properly
TEST_F(ParametersTest, testDefaultParameterValues)
{
  EXPECT_EQ(ONE_CLASS,paramInst->getSvmType());
  EXPECT_EQ(RBF,paramInst->getKernelType());
  EXPECT_EQ(0,paramInst->getDegree());
  EXPECT_EQ(0,paramInst->getGamma());
  EXPECT_EQ(0,paramInst->getCoef());
  EXPECT_EQ(200,paramInst->getCacheSize());
  EXPECT_EQ(0.001,paramInst->getEps());
  EXPECT_EQ(0.1,paramInst->getC());
  EXPECT_EQ(0,paramInst->getNrWeight());
  EXPECT_EQ(-1,paramInst->getWeightLabel()[0]);
  EXPECT_EQ(1,paramInst->getWeightLabel()[1]);
  EXPECT_EQ(-1,paramInst->getWeight()[0]);
  EXPECT_EQ(1,paramInst->getWeight()[1]);
  EXPECT_EQ(0.5,paramInst->getNu());
  EXPECT_EQ(0,paramInst->getP());
  EXPECT_EQ(0,paramInst->getShrinking());
  EXPECT_EQ(0,paramInst->getProbability());
}

/// @brief Test that the setter and getter methods function properly
TEST_F(ParametersTest, testSetters)
{
  paramInst->setSvmType(1);
  EXPECT_EQ(1,paramInst->getSvmType());

  paramInst->setKernelType(1);
  EXPECT_EQ(1,paramInst->getKernelType());

  paramInst->setDegree(1);
  EXPECT_EQ(1,paramInst->getDegree());

  paramInst->setGamma(0.01);
  EXPECT_EQ(0.01,paramInst->getGamma());

  paramInst->setCoef(0.01);
  EXPECT_EQ(0.01,paramInst->getCoef());

  paramInst->setCacheSize(100);
  EXPECT_EQ(100,paramInst->getCacheSize());

  paramInst->setEps(0.005);
  EXPECT_EQ(0.005,paramInst->getEps());

  paramInst->setC(0.2);
  EXPECT_EQ(0.2,paramInst->getC());

  paramInst->setNrWeight(1);
  EXPECT_EQ(1,paramInst->getNrWeight());

  paramInst->setNu(1);
  EXPECT_EQ(1,paramInst->getNu());

  paramInst->setP(1);
  EXPECT_EQ(1,paramInst->getP());

  paramInst->setShrinking(1);
  EXPECT_EQ(1,paramInst->getShrinking());

  paramInst->setProbability(1);
  EXPECT_EQ(1,paramInst->getProbability());
}
}

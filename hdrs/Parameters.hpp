#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <mutex.hpp>
#include "svm.h"

/// @file
/// @brief Interface used to store and retrieve application parameters

/// The Parameters class is used to store and retrieve the model parameters as well as parameters used when training a model
class Parameters
{
public:
  /// @brief Returns the single instance of Parameters
  static Parameters* instance();

  /// @brief Sets the type of svm model to build
  /// @param newVal
  ///        The SVM model type: C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
  void setSvmType(int newVal)
  {
    paramsMutex.lock();
    svmType = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the type of kernel to use 
  /// @param newVal
  ///        The kernel type: LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
  void setKernelType(int newVal)
  {
    paramsMutex.lock();
    kernelType = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the highest degree of a polynomial kernel function
  /// @param newVal
  void setDegree(int newVal)
  {
    paramsMutex.lock();
    degree = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the gamma value used in the kernel function
  /// @param newVal
  ///        Gamma value used in polynomial, RBF and sigmoid kernel functions
  void setGamma(double newVal)
  {
    paramsMutex.lock();
    gamma = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the coefficient used in the kernel function
  /// @param newVal
  ///        Coefficient used in polynomial and sigmoid kernel functions
  void setCoef(double newVal)
  {
    paramsMutex.lock();
    coef0 = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the initial cache size used when creating the model
  /// @param newVal
  ///        The unit of the new value is in Mbytes 
  void setCacheSize(double newVal)
  {
    paramsMutex.lock();
    cacheSize = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the epsilon value used in the loss function when creating regression models
  /// @param newVal
  void setEps(double newVal)
  {
    paramsMutex.lock();
    eps = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the cost parameter used when determining the model parameters
  /// @param newVal
  ///        The C value used when training the C-SVC, EPSILON-SVR and NU-SVR SVM models
  void setC(double newVal)
  {
    paramsMutex.lock();
    C = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the number of penalty weights used when building classification models
  /// @param newVal
  ///        Number of weight labels used when training the C-SVC model
  void setNrWeight(int newVal)
  {
    paramsMutex.lock();
    nrWeight = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the nu value used to approximate the fraction of training errors and support vectors
  /// @param newVal
  ///        Nu value used when training the NU_SVC, ONE_CLASS and NU_SVR models
  void setNu(double newVal)
  {
    paramsMutex.lock();
    nu = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the P value used when training an epsilon support vector regression model
  /// @param newVal
  ///        P value used when training the EPSILON_SVR model
  void setP(double newVal)
  {
    paramsMutex.lock();
    p = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the parameter used to determine if a shrinking heuristic should be applied during training
  /// @param newVal
  void setShrinking(int newVal)
  {
    paramsMutex.lock();
    shrinking = newVal;
    paramsMutex.unlock();
  }

  /// @brief Sets the parameters used to deterimine if probability estimates should be applied during training
  /// @param newVal
  void setProbability(int newVal)
  {
    paramsMutex.lock();
    probability = newVal;
    paramsMutex.unlock();
  }

  /// @brief Returns the type of svm model to build
  int getSvmType()
  {
    paramsMutex.lock();
    int retValue = svmType;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the type of kernel to use 
  int getKernelType()
  {
    paramsMutex.lock();
    int retValue = kernelType;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the degree of the kernel function
  int getDegree()
  {
    paramsMutex.lock();
    int retValue = degree;
    paramsMutex.unlock();

    return retValue;
  }     

  /// @brief Returns the gamma value used in the kernel function
  double getGamma()
  {
    paramsMutex.lock();
    double retValue = gamma;
    paramsMutex.unlock();

    return retValue;
  }   

  /// @brief Returns the coefficient used in the kernel function
  double getCoef()
  {
    paramsMutex.lock();
    double retValue = coef0;
    paramsMutex.unlock();

    return retValue;
  }   

  /// @brief Returns the initial cache size used when creating the model
  double getCacheSize()
  {
    paramsMutex.lock();
    double retValue = cacheSize;
    paramsMutex.unlock();

    return retValue;
  } 

  /// @brief Returns the epsilon value used in the loss function when creating regression models
  double getEps()
  {
    paramsMutex.lock();
    double retValue = eps;
    paramsMutex.unlock();

    return retValue;
  }     

  /// @brief Returns the cost parameter used when determining the model parameters
  double getC()
  {
    paramsMutex.lock();
    double retValue = C;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the number of penalty weights used when building classification models
  int getNrWeight()          
  {
    paramsMutex.lock();
  double retValue = nrWeight;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the weight labels used for the model being trained
  int* getWeightLabel()      
  {
    paramsMutex.lock();
    int* retValue = weightLabel;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the weight values used for the model being trained
  double* getWeight()
  {
    paramsMutex.lock();
    double* retValue = weight;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the nu value used to approximate the fraction of training errors and support vectors
  double getNu()      
  {
    paramsMutex.lock();
    double retValue = nu;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the P value used when training an epsilon support vector regression model
  double getP()       
  {
    paramsMutex.lock();
    double retValue = p;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the parameter used to determine if a shrinking heuristic should be applied during training
  int getShrinking()  
  {
    paramsMutex.lock();
    double retValue = shrinking;
    paramsMutex.unlock();

    return retValue;
  }

  /// @brief Returns the parameters used to deterimine if probability estimates should be applied during training
  int getProbability()
  {
    paramsMutex.lock();
    double retValue = probability;
    paramsMutex.unlock();
  
    return retValue;
  } 

protected:

  /// Default constructor
  Parameters()
  {
    paramsMutex.lock();
    svmType = ONE_CLASS;
    kernelType = RBF;
    degree = 0;     
    gamma = 0;   
    coef0 = 0;   
    cacheSize = 200; 
    eps = 0.001;     
    C = 0.1;       
    nrWeight = 0;          
    weightLabel = new int[2];      
    weightLabel[0] = -1;
    weightLabel[1] = 1;
    weight = new double[2];         
    weight[0] = -1;
    weight[1] = 1;
    nu = 0.5;      
    p = 0;       
    shrinking = 0;  
    probability = 0; 
    paramsMutex.unlock();
  }

  /// Destructor
  ~Parameters()
  {
    if (NULL != Parameters::inst)
    {
      delete weightLabel;
      delete weight;
      delete Parameters::inst;
    }
  }

  /// Singleton instance
  static Parameters* inst; 

  /// Mutex used to ensure that gui which may be running
  /// does not create race condition
  Mutex paramsMutex;       

  /// The SVM model type. Valid values are C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
  int svmType;

  /// The kernel type. Value values are LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
  int kernelType;

  /// The highest degree of a polynomial kernel function  
  int degree;

  /// Gamma value used for Polynomial, RBF and sigmoid kernel functions
  double gamma;   

  /// Coefficient used in polynomial and sigmoid functions
  double coef0;   

  /// Initial cache size allocated when training an SVM model
  double cacheSize; 

  /// Epsilong value used as stopping criteria during training
  double eps;     

  /// C value used when training C_SVC, EPSILON_SVR and NU_SVR models
  double C;       

  /// The number of weight labels used when training an SVM model
  int nrWeight;          

  /// The labels that the weights will be associated with.
  int *weightLabel;      

  /// The weights associated with the labels.
  double* weight;         

  /// The nu value used when training NU_SVC, ONE_CLASS or NU_SVR models
  double nu;      

  /// The p value used when training EPSILON_SVR models
  double p;       

  /// Value describing whether shrinking heuristics should be used when training
  int shrinking;  

  /// Value used when probability estimates should be used when training
  int probability; 
};

#endif

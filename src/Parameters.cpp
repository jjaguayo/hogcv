#include "Parameters.hpp"

Parameters* Parameters::inst = NULL;

Parameters* Parameters::instance()
{
  if (NULL == Parameters::inst)
  {
    Parameters::inst = new Parameters();
  }
  return Parameters::inst;
}

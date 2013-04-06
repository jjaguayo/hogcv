#include "SetParameters.hpp"
#include <iostream>

SetParametersDialog::SetParametersDialog(Parameters* params, QWidget *parent) : QDialog(parent)
{
  setupUi(this);

  cannyCheckBox->setChecked(params->getCanny());
  cannyThresh1Spin->setValue(params->getCannyThresh1());
  cannyThresh2Spin->setValue(params->getCannyThresh2());
  linearTransCheckBox->setChecked(params->getLinearTrans());
  alphaSlider->setValue(params->getAlpha());
  betaSlider->setValue(params->getBeta());
  grayCheckBox->setChecked(params->getTransformToGray());
  gammaCheckBox->setChecked(params->getGammaCorrection()); 
  gammaSlider->setValue((int)params->getGammaValue()*10.0);
  hist1CheckBox->setChecked(params->getComputeHist());
  hist2CheckBox->setChecked(params->getUseHistFunction()); 

  connect(SetButton,SIGNAL(clicked()),this,SLOT(setButtonClicked()));
}

void SetParametersDialog::setButtonClicked()
{
  Parameters::instance()->setCanny(cannyCheckBox->isChecked());
  Parameters::instance()->setCannyThresh1(cannyThresh1Spin->value());
  Parameters::instance()->setCannyThresh2(cannyThresh2Spin->value());

  Parameters::instance()->setLinearTrans(linearTransCheckBox->isChecked());
  Parameters::instance()->setAlpha(alphaSlider->value());
  Parameters::instance()->setBeta(betaSlider->value());
  Parameters::instance()->setTransformToGray(grayCheckBox->isChecked());
  Parameters::instance()->setGammaCorrection(gammaCheckBox->isChecked());
  Parameters::instance()->setGammaValue(gammaSlider->value()/10.0);
  Parameters::instance()->setComputeHist(hist1CheckBox->isChecked());
  Parameters::instance()->setUseHistFunction(hist2CheckBox->isChecked());
}

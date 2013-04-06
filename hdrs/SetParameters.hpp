#ifndef SETPARAM_H
#define SETPARAM_H

#include <QDialog>
#include <ui_SetParameters.h>
#include "Parameters.hpp"

/// @file
/// @brief Dialog box to set the parameters for the SVM model. 
/// The functionality is yet to be implemented.

/// @brief Dialog box used to set the parameters used by the application
/// to train the model and classify images.
class SetParametersDialog : public QDialog, public Ui::SetDialog
{
  Q_OBJECT

public:
  /// @brief Constructor
  SetParametersDialog(Parameters* param, QWidget *parent = 0);

private slots:

  /// @brief Slot used to handle buttonClicked events
  void setButtonClicked();
};
#endif

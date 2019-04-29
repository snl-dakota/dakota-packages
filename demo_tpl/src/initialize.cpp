/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright 2014 Sandia Corporation.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       namespaced free function
//- Description: Demo TPL initialize
//- Owner:       Russell Hooper
//- Checked by:  ...

#include<iostream>

// Demo TPL headers
#include "initialize.hpp"

bool
Demo_TPL::initialize(bool verbose)
{

  if( verbose )
    std::cout << "Doing Demo_TPL::initialize." << std::endl;

  return true;
}

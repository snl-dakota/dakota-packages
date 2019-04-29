/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright 2014 Sandia Corporation.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       namespaced free function
//- Description: Demo TPL execute
//- Owner:       Russell Hooper
//- Checked by:  ...

#include<iostream>

// Demo TPL headers
#include "execute.hpp"

bool
Demo_TPL::execute(bool verbose)
{

  if( verbose )
    std::cout << "Doing Demo_TPL::execute." << std::endl;

  return true;
}

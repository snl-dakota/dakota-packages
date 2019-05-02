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
#include "demo_opt.hpp"

// -----------------------------------------------------------------

bool
Demo_Opt::set_solver_options(const std::string & filename, bool verbose)
{
  if( verbose )
    std::cout << "Setting Demo_Opt solver options using file \""<<filename<<"\"" << std::endl;

  options_file_ = filename;

  return true;
}

// -----------------------------------------------------------------

bool
Demo_Opt::initialize(bool verbose)
{

  if( verbose )
    std::cout << "Doing Demo_TPL::initialize." << std::endl;

  return true;
}

// -----------------------------------------------------------------

bool
Demo_Opt::execute(bool verbose)
{

  if( verbose )
    std::cout << "Doing Demo_TPL::execute." << std::endl;

  return true;
}

// -----------------------------------------------------------------

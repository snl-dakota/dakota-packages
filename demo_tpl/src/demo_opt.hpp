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

/** ... description ...
 * */

#ifndef DEMO_INITIALIZE_H
#define DEMO_INITIALIZE_H

#include <string>
class Demo_Opt
{
  public:

    // Default ctor
    Demo_Opt() {}

    // Allow specification of options filename
    bool set_solver_options(const std::string & filename, bool verbose = false);

    // A simple initialization method
    bool initialize(bool verbose = false);

    // A simple initialization method
    bool execute(bool verbose = false);


  private:

    std::string options_file_;
};

#endif

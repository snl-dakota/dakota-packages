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
#include <map>

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

    // Parameter settings
    template< typename T >
      void set_param(const std::string & param, const T & val)
        { set_parameter_value(param, val); }



  private:

    void set_parameter_value(const std::string & param, const int & val)
      { int_params_[param] = val; }
    void set_parameter_value(const std::string & param, const double & val)
      { dbl_params_[param] = val; }

    std::string options_file_;

    std::map<std::string, int>    int_params_;
    std::map<std::string, double> dbl_params_;

};

#endif

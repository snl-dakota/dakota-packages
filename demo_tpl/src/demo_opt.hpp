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

    enum class SOLVE_OPTIONS_INT { MAX_EVALS,
                                   MAX_ITERS };

    enum class SOLVE_OPTIONS_DBL { CONV_TOL,
                                   OBJ_TARGET };

    // Default ctor
    Demo_Opt() {}

    // Allow specification of options filename
    bool set_solver_options(const std::string & filename, bool verbose = false);

    // A simple initialization method
    bool initialize(bool verbose = false);

    // A simple initialization method
    bool execute(bool verbose = false);

    // Parameter settings
    template< typename Enum_T, typename T >
      void set_param(Enum_T etype, const T & val)
        { set_parameter_value(etype, val); }



  private:

    void set_parameter_value(SOLVE_OPTIONS_INT ieval, const int & val)
      { int_params_[ieval] = val; }
    void set_parameter_value(SOLVE_OPTIONS_DBL deval, const double & val)
      { dbl_params_[deval] = val; }

    std::string options_file_;

    std::map<SOLVE_OPTIONS_INT, int>    int_params_;
    std::map<SOLVE_OPTIONS_DBL, double> dbl_params_;

};

#endif

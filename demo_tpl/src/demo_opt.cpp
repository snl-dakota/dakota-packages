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

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>

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
  {
    std::cout << "Doing Demo_Opt::initialize." << std::endl;
    std::cout << "Registered parameters :\n";
    for( auto ip : int_params_ )
      std::cout << ip.first << " = " << ip.second << std::endl;
    for( auto dp : dbl_params_ )
      std::cout << dp.first << " = " << dp.second << std::endl;
  }

  return true;
}

// -----------------------------------------------------------------

void
Demo_Opt::set_problem_data(const std::vector<double> & init ,
                           const std::vector<double> & lower,
                           const std::vector<double> & upper )
{
  assert( init.size() == lower.size() );
  assert( init.size() == upper.size() );

  init_vals_  = init;
  lower_bnds_ = lower;
  upper_bnds_ = upper;
}

// -----------------------------------------------------------------

bool
Demo_Opt::execute(bool verbose)
{
  if( verbose )
    std::cout << "Doing Demo_Opt::execute." << std::endl;

  assert( int_params_.count("Maximum Evaluations") > 0 );
  int max_evals = int_params_["Maximum Evaluations"];

  int num_params = (int)init_vals_.size();
  best_x_.clear();
  best_f_ = std::numeric_limits<double>::max();

  //assert( dbl_params_.count("Objective Target") > 0 );
  //double target = int_params_["Objective Target"];
  double target = 0.0; // based on the SimpleQuadratic

  std::default_random_engine generator;
  std::vector< std::uniform_real_distribution<double> > distributions;
  for( size_t i=0; i<init_vals_.size(); ++i )
    distributions.push_back(std::uniform_real_distribution<double>(lower_bnds_[i],upper_bnds_[i]));

  // Hard-coded to a single parameter for now...
  std::vector<double> x(num_params);
  double fn;
  for( int i=0; i<=max_evals; ++i )
  {
    for( int np=0; np<num_params; ++np )
      x[i] = distributions[i](generator);
    fn = obj_fn_callback_->compute_obj(x, false);
    if( fabs(fn-target) < best_f_ )
    {
      best_x_ = x;
      best_f_ = fabs(fn-target);
    }
  }

  if( verbose )
    std::cout << "Found best_f_ = " << best_f_ << std::endl;

  return true;
}

// -----------------------------------------------------------------

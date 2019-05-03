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

  best_x_.clear();
  best_f_ = std::numeric_limits<double>::max();

  //assert( dbl_params_.count("Objective Target") > 0 );
  //double target = int_params_["Objective Target"];
  double target = 0.0; // based on the SimpleQuadratic

  int num_samples = 2+static_cast<int>(pow(max_evals, 1.0/init_vals_.size()));
  std::vector<double> dp(init_vals_.size());
  for( size_t i=0; i<init_vals_.size(); ++i )
    dp[i] = (upper_bnds_[i] - lower_bnds_[i])/num_samples;
  //std::cout << "I'd like to perform " << num_samples << " in each of " << dp.size() << "dimensions." << std::endl;

  // Hard-coded to a single parameter for now...
  double x, fn, best_x;
  for( int i=0; i<=num_samples; ++i )
  {
    x = lower_bnds_[0] + i*dp[0];
    fn = obj_fn_callback_->compute_obj(x, false);
    if( fabs(fn-target) < best_f_ )
    {
      best_x = x;
      best_f_ = fabs(fn-target);
    }
  }

  if( verbose )
    std::cout << "Found best_x = " << best_x << " with best_f_ = " << best_f_ << std::endl;

  best_x_.push_back(best_x);
  for( size_t i=1; i<init_vals_.size(); ++i )
    best_x_.push_back(0.0); // need to fix this for multiple params

  return true;
}

// -----------------------------------------------------------------

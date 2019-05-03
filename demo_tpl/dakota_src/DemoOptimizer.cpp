/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright 2014 Sandia Corporation.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       DemoTPLOptimizer
//- Description: Wrapper class for Demo_Opt
//- Owner:       Russell Hooper
//- Checked by:  

// Dakota headers
#include "DemoOptimizer.hpp"
#include "ProblemDescDB.hpp"

// Demo_Opt headers
#include "demo_opt.hpp"

//
// - DemoTPLOptimizer implementation
//

namespace Dakota {

  class SimpleQuadratic
  {
    public :

      SimpleQuadratic(const Real k=1.0) :
        kVal(k)
    { }

      Real compute_obj(const Real x)
      {
        return 0.5*(kVal-x)*(kVal-x);
      }

    private:

      Real kVal;
  };

// -----------------------------------------------------------------
/** Implementation of DemoTPLOptimizer class. */


// Standard constructor for DemoTPLOptimizer.  Sets up Demo_Opt solver based on
// information from the problem database.
DemoTPLOptimizer::DemoTPLOptimizer(ProblemDescDB& problem_db, Model& model):
  Optimizer(problem_db, model, std::shared_ptr<TraitsBase>(new DemoOptTraits())),
  Demo_Opt::ObjectiveFn(),
  demoOpt(std::make_shared<Demo_Opt>())
{
  set_demo_parameters();

  // Register ouself as the callback interface for objective fn evaluations
  demoOpt->register_obj_fn(this);
}


// -----------------------------------------------------------------

// core_run redefines the Optimizer virtual function to perform the
// optimization using Demo_Opt and catalogue the results.
void DemoTPLOptimizer::core_run()
{
  initialize_variables_and_constraints();

  demoOpt->execute(true);

  if (!localObjectiveRecast) {
    double best_f = demoOpt->get_best_f();

    const BoolDeque& max_sense = iteratedModel.primary_response_fn_sense();
    RealVector best_fns(numFunctions);
    best_fns[0] = (!max_sense.empty() && max_sense[0]) ?
      -best_f : best_f;
    bestResponseArray.front().function_values(best_fns);
  }

  std::vector<double> best_x = demoOpt->get_best_x();
  set_variables< std::vector<double> >(best_x, iteratedModel, bestVariablesArray.front());

} // core_run


// -----------------------------------------------------------------

void DemoTPLOptimizer::initialize_run()
{
  Optimizer::initialize_run();
  demoOpt->initialize(true);
}


// -----------------------------------------------------------------

void DemoTPLOptimizer::set_demo_parameters()
{
  int max_evaluations
    = probDescDB.get_int("method.max_function_evaluations");
  demoOpt->set_param("Maximum Evaluations", max_evaluations);

  int max_iterations
    = probDescDB.get_int("method.max_iterations");
  demoOpt->set_param("Maximum Iterations", max_iterations);

  const Real& min_f_change
    = probDescDB.get_real("method.convergence_tolerance");
  demoOpt->set_param("Function Tolerance", min_f_change);

  const Real& min_var_change
    = probDescDB.get_real("method.variable_tolerance");
  demoOpt->set_param("Step Tolerance", min_var_change);

  const Real& objective_target
    = probDescDB.get_real("method.solution_target");
  demoOpt->set_param("Objective Target", objective_target);


  // Check for native Demo_Opt input file.
  String adv_opts_file = probDescDB.get_string("method.advanced_options_file");
  if (!adv_opts_file.empty())
  {
    if (!boost::filesystem::exists(adv_opts_file))
    {
      Cerr << "\nError: Demo_Opt options_file '" << adv_opts_file
	   << "' specified, but file not found.\n";
      abort_handler(METHOD_ERROR);
    }
  }

  demoOpt->set_solver_options(adv_opts_file, true);

} // set_demo_parameters

// -----------------------------------------------------------------

void DemoTPLOptimizer::initialize_variables_and_constraints()
{

  // just do continuous variables; use iteratedModel method to get number
  // of variables rather than internal Dakota variable names
  int num_total_vars = numContinuousVars;
  std::vector<Real> init_point(num_total_vars);
  std::vector<Real> lower(num_total_vars),
                    upper(num_total_vars);

  // need traits; just do bounds for now, not linear/nonlinear
  get_variables(iteratedModel, init_point);
  get_variable_bounds_from_dakota<DemoOptTraits>( lower, upper );

  demoOpt->set_problem_data(init_point,   //  "Initial Guess"
                            lower     ,   //  "Lower Bounds"
                            upper      ); //  "Upper Bounds"

} // initialize_variables_and_constraints

// -----------------------------------------------------------------

void DemoTPLOptimizer::evaluate_function(const std::vector<double> x,
					 double& f)
{
  set_variables<std::vector<double> >(x, iteratedModel, iteratedModel.current_variables());

  iteratedModel.evaluate();// default active s
  const BoolDeque& max_sense = iteratedModel.primary_response_fn_sense();

  f = (!max_sense.empty() && max_sense[0]) ?
      -iteratedModel.current_response().function_value(0) :
       iteratedModel.current_response().function_value(0);

} // evaluate_function

// -----------------------------------------------------------------

Real
DemoTPLOptimizer::compute_obj(const double & x, bool verbose) const
{
  SimpleQuadratic obj;
  Real val = obj.compute_obj(x);
  if( verbose )
    Cout << "DemoTPLOptimizer::compute_obj: F("<<x<<") = " << val << std::endl;

  return val;
}

} // namespace Dakota

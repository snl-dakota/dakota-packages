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

// -----------------------------------------------------------------
/** Implementation of DemoTPLOptimizer class. */


// Standard constructor for DemoTPLOptimizer.  Sets up Demo_Opt solver based on
// information from the problem database.
DemoTPLOptimizer::DemoTPLOptimizer(ProblemDescDB& problem_db, Model& model):
  Optimizer(problem_db, model, std::shared_ptr<TraitsBase>(new DemoOptTraits())),
  demoOpt(std::make_shared<Demo_Opt>())
{
  set_demo_parameters();
}


// -----------------------------------------------------------------

// core_run redefines the Optimizer virtual function to perform the
// optimization using Demo_Opt and catalogue the results.
void DemoTPLOptimizer::core_run()
{
  demoOpt->execute(true);
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

} // namespace Dakota

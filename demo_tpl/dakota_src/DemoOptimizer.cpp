/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright 2014 Sandia Corporation.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       DemoTPLOptimizer
//- Description: Wrapper class for DemoTPL
//- Owner:       Russell Hooper
//- Checked by:  

// Dakota headers
#include "DemoOptimizer.hpp"
#include "ProblemDescDB.hpp"

// DemoTPL headers
#include "initialize.hpp"
#include "execute.hpp"

//
// - DemoTPLOptimizer implementation
//

namespace Dakota {

// -----------------------------------------------------------------
/** Implementation of DemoTPLOptimizer class. */


// Standard constructor for DemoTPLOptimizer.  Sets up DemoTPL solver based on
// information from the problem database.
DemoTPLOptimizer::DemoTPLOptimizer(ProblemDescDB& problem_db, Model& model):
  Optimizer(problem_db, model, std::shared_ptr<TraitsBase>(new DemoTPLTraits()))
{
  /* no-op */
}



// core_run redefines the Optimizer virtual function to perform the
// optimization using DemoTPL and catalogue the results.
void DemoTPLOptimizer::core_run()
{
  Demo_TPL::execute(true);
} // core_run



void DemoTPLOptimizer::initialize_run()
{
  Optimizer::initialize_run();
  Demo_TPL::initialize(true);
}

} // namespace Dakota

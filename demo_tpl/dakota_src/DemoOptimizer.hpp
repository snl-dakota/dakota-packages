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

/** description...
 * */

#ifndef DEMOTPL_OPTIMIZER_H
#define DEMOTPL_OPTIMIZER_H

// Dakota headers
#include "DakotaOptimizer.hpp"
//#include "DakotaModel.hpp"
//#include "DakotaTraitsBase.hpp"

// DemoTPL headers
//#include "initialize.hpp"
//#include "execute.hpp"

namespace Dakota {

// -----------------------------------------------------------------
/** DemoTPLOptimizer specializes DakotaOptimizer to show the steps needed
 *  to construct and run a DemoTPL solver */

class DemoTPLOptimizer : public Optimizer
{
public:

  //
  //- Heading: Constructors and destructor
  //

  /// Standard constructor
  DemoTPLOptimizer(ProblemDescDB& problem_db, Model& model);

  /// Destructor
  ~DemoTPLOptimizer() {}

  //
  //- Heading: Virtual member function redefinitions
  //

  /// Initializes the DemoTPL optimizer
  void initialize_run() override;

  /// Executes the DemoTPL solver
  void core_run() override;

protected:

}; // class DemoTPLOptimizer


// -----------------------------------------------------------------
/** DemoTPLTraits defines the types of problems and data formats DemoTPL
    supports by overriding the default traits accessors in
    TraitsBase. */

class DemoTPLTraits: public TraitsBase
{
public:

  //
  //- Heading: Constructor and destructor
  //

  /// Default constructor
  DemoTPLTraits() { }

  /// Destructor
  virtual ~DemoTPLTraits() { }

  /// DemoTPL default data type to be used by Dakota data adapters
  typedef std::vector<Real> VecT;

  //
  //- Heading: Virtual member function redefinitions
  //

  // By default we do not support anything

  /// Return flag indicating support for continuous variables
  //bool supports_continuous_variables() { return true; }

}; // class DemoTPLTraits

} // namespace Dakota

#endif

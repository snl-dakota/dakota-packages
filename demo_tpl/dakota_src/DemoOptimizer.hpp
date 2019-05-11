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

/** description...
 * */

#ifndef DEMOTPL_OPTIMIZER_H
#define DEMOTPL_OPTIMIZER_H

// Dakota headers
#include "DakotaOptimizer.hpp"

// Demo_Opt TPL headers
#include "demo_opt.hpp"

namespace Dakota {

// -----------------------------------------------------------------
/** DemoTPLOptimizer specializes DakotaOptimizer to show the steps needed
 *  to construct and run a Demo_Opt solver */

class DemoTPLOptimizer : public Optimizer,
                         public Demo_Opt::ObjectiveFn
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

    /// Initializes the Demo_Opt optimizer
    void initialize_run() override;

    /// Executes the Demo_Opt solver
    void core_run() override;

    /// Inherits Demo_TPL::ObjectiveFn
    Real compute_obj(const std::vector<double> & x, bool verbose) override;

  protected:

  /// Helper functions to pass Dakota set-up data to TPL.
  void set_demo_parameters();
  void initialize_variables_and_constraints();

  /// Instaintiation of demo optimizer object.
  std::shared_ptr<Demo_Opt> demoOpt;

}; // class DemoTPLOptimizer


// -----------------------------------------------------------------
/** Traits are the means by which information about what the TPL
    supports is communicated to Dakota for internal use.
    DemoOptTraits defines the types of problems and data formats
    Demo_Opt supports by overriding the default traits accessors in
    TraitsBase.  By default, nothing is supported, and the TPL
    integrator must explicitly turn on the traits for any supported
    features.  This simple example only supports continuous variables.
    Examples showing other features will be added in future updates to
    this demo. */

class DemoOptTraits: public TraitsBase
{
public:

  //
  //- Heading: Constructor and destructor
  //

  /// Default constructor
  DemoOptTraits() { }

  /// Destructor
  virtual ~DemoOptTraits() { }

  /// Demo_Opt default data type to be used by Dakota data adapters
  typedef std::vector<double> VecT;

  /// This is needed to handle constraints
  inline static double noValue()
  { return std::numeric_limits<Real>::max(); }

  //
  //- Heading: Virtual member function redefinitions
  //

  /// Return the flag indicating whether method supports continuous variables
  bool supports_continuous_variables() { return true; }

}; // class DemoOptTraits

} // namespace Dakota

#endif

Demo TPL        {#demotpl}
========

This is a simple Demo which serves as a working example for bringing a
new Third-Party-Library (TPL) into Dakota.  The Demo will serve to
show minimal requirements for:

 - building the Demo library under Dakota via Cmake
 - exposing Demo functionality, eg initialization and execution, to Dakota
 - exposing Demo options to Dakota
 - transferring data, variables and responses, between Demo and Dakota

Following this Demo, a developer should be able to integrate an
optimization TPL/method that is derivative-free, operates over
continuous variables, and supports bound constraints.


## Building _Demo_ under Dakota using Cmake

This section shows how to include the relevant parts of the `Demo` TPL
as a library that Dakota builds and includes as part of its own native
Cmake build.

Assuming the _Demo_ tpl source code has been placed alongside other
Dakota TPLs in `$DAKTOA_SRC/packages/external/demo_tpl`, a simple
_CMakeLists.txt_ file can be created at this location to allow Dakota to
include it within its own Cmake setup.  An minimal example might include:
 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cmake}
# File $DAKTOA_SRC/packages/external/demo_tpl/CMakeLists.txt

cmake_minimum_required(VERSION 2.8)
project("DEMO_TPL" CXX)
SUBDIRS(src)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In the src subdirectory of demo_tpl would be another _CMakeLists.txt_
file which essentially identifies the relevant source code to be
compiled into a library along with defining the library which Daktoa
will later include, eg

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cmake}
# File $DAKTOA_SRC/packages/external/demo_tpl/src/CMakeLists.txt

set(demo_tpl_HEADERS
    demo_opt.hpp
   )

set(demo_tpl_SOURCES
    demo_opt.cpp
   )

# Set the DEMO_TPL library name.
add_library(demo_tpl ${demo_tpl_SOURCES})

# Define install targets for "make install"
install(TARGETS demo_tpl EXPORT ${ExportTarget} DESTINATION lib)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Note that it is possible to use Cmake's glob feature to bring in all
source and header files, but care must be taken to avoid introducing
`main(...)` symbols which will collide with Dakota's `main` at link
time.


At this point, Dakota's _CMakeLists.txt_ files will need to be
modified to include the _Demo_ tpl library.  The following modified
can be used to bring in the _Demo_ TPL conditioned on having `-D
HAVE_DEMO_TPL:BOOL=ON` defined when invoking cmake to configure
Dakota:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cmake}
# File $DAKTOA_SRC/packages/CMakeLists.txt

<... snip ...>
  option(HAVE_DEMO_TPL "Build the Demo_TPL package." OFF)
<... end snip ...>

<... snip ...>
  if(HAVE_DEMO_TPL)
    add_subdirectory(external/demo_tpl)
  endif(HAVE_DEMO_TPL)
<... end snip ...>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This next modification to Dakota will allow the _Demo_ TPL to be used
by other Dakota source code by including the necessary include paths,
link-time libraries and needed #defines:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cmake}
# File $DAKTOA_SRC/src/CMakeLists.txt

<... snip ...>

if(HAVE_DEMO_TPL)
  set(DAKOTA_DEMOTPL_ROOT_DIR "${Dakota_SOURCE_DIR}/packages/external/demo_tpl")
  list(APPEND DAKOTA_INCDIRS 
      ${DAKOTA_DEMOTPL_ROOT_DIR}/dakota_src
      ${DAKOTA_DEMOTPL_ROOT_DIR}/src)
set(iterator_src ${iterator_src} ${Dakota_SOURCE_DIR}/packages/external/demo_tpl/dakota_src/DemoOptimizer.cpp)
  list(APPEND DAKOTA_PKG_LIBS demo_tpl)
list(APPEND EXPORT_TARGETS demo_tpl)
  add_definitions("-DHAVE_DEMO_TPL")
endif(HAVE_DEMO_TPL)

<... end snip ...>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
## Modifying Dakota to use the _Demo_  TPL

 Before making concrete changes, it is often helpful to create a
 simple Dakota test which will serve to guide the process.  This is
 akin to test-driven development which essentially creates a test
 which fails until everything has been implemented to allow it to run
 and pass. An candidate test for the current activity could be the
 following:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
# File $DAKTOA_SRC/test/dakota_demo_tpl.in

    method,
        demo_tpl
        options_file = "demo_tpl.opts"

    variables,
        continuous_design = 3
        initial_point      -1.0    1.5   2.0
        upper_bounds	   10.0   10.0  10.0
        lower_bounds       -10.0  -10.0 -10.0
        descriptors	    'x1'  'x2'  'x3'

    interface,
        direct
        analysis_driver = 'text_book'

    responses,
        objective_functions = 1
        no_gradients
        no_hessians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


For this test to run, we will need to be able to pass parsed options
to the _Demo_ TPL and exchange parameters and response values between
Dakota and _Demo_ TPL.  These details are presented in the following
two sections.

## Exchanging Parameters and Reponses

Like any TPL, the _Demo_ TPL will need to exchange parameter and
obective function values with Dakota.  For purposes of demonstration,
an example interface between Dakota and the _Demo_ TPL can be seen in
$DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.hpp (with
corresponding .cpp in the same directory).  Within these files is a
key callback interface used by the _Demo_ TPL to obtain objective
function values for given parater values (3 in the test above), eg:


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// File $DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.cpp

Real
DemoTPLOptimizer::compute_obj(const std::vector<double> & x, bool verbose)
{
  set_variables<std::vector<double> >(x, iteratedModel, iteratedModel.current_variables());

  iteratedModel.evaluate();
  const BoolDeque& max_sense = iteratedModel.primary_response_fn_sense();

  double f = (!max_sense.empty() && max_sense[0]) ?
    -iteratedModel.current_response().function_value(0) :
    iteratedModel.current_response().function_value(0);

  return f;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In this instance, the _Demo_ TPL uses `std::vector<double>` as its
native parameter vector data type and is calling back to the example
problem via an interface to Dakota to obtain a single `double`
(aliased to `Real` in Dakota) obective function value for a given set
of parameter values.  These data exchanges are facilitated by used of
"data adapters" supplied by Dakota with the `set_variables<>(...)`
helper utilized in this case.

Similarly, Dakota must provide initial parameter values to the TPL
and retrieve final objective function and variable values from the
TPL.  The initial values for parameters and bound constraints can be
obtained from Dakota with the `get_variables<>(...)` helpers.  This
example returns the values to a standard vector of doubles (Reals).
Those values can then be passed on the TPL using TPL API.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// File $DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.cpp

int num_total_vars = numContinuousVars;
std::vector<Real> init_point(num_total_vars);
std::vector<Real> lower(num_total_vars),
                  upper(num_total_vars);

get_variables(iteratedModel, init_point);
get_variable_bounds_from_dakota<DemoOptTraits>( lower, upper );

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The TPL should be able to return and optimal objective function value
and the corresponding variable values via its API.  As has been the
case throughout, the data should be doubles.  The following code take
the returned values and set the Dakota data structures that contain
final objective and variable values.  It adjusts the sign of the
objective based on whether minimize or maximize has been specified in
the Dakota input file (minimize is the default).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// File $DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.cpp

  double best_f = demoOpt->get_best_f();

  const BoolDeque& max_sense = iteratedModel.primary_response_fn_sense();
  RealVector best_fns(numFunctions);
  best_fns[0] = (!max_sense.empty() && max_sense[0]) ?
    -best_f : best_f;
  bestResponseArray.front().function_values(best_fns);
}

std::vector<double> best_x = demoOpt->get_best_x();

set_variables< std::vector<double> >(best_x, iteratedModel, bestVariablesArray.front());
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


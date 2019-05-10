
This is a simple Demo which serves as a working example for bringing a new 
Third-Party-Library (TPL) into Dakota.  The Demo will serve to show minimal
requirements for:

 - building the Demo library under Dakota via Cmake
 - exposing Demo functionality, eg initialzation and execution, to Dakota
 - exposing Demo options to Dakota
 - transferring data, variables and responses, between Demo and Dakota

# Building _Demo_ under Dakota using Cmake

 This section shows how to include the relevant parts of the `Demo` TPL as a library 
 that Dakota builds and includes as part of its own native Cmake build.

 Assuming the _Demo_ tpl source code has been placed alongside other Dakota TPLs in
 `$DAKTOA_SRC/packages/external/demo_tpl`, a simple _CMakeLists.txt_ file can be created
 at this location to allow Dakota to include it within its own Cmake setup.  An minimal
 example might include:
 
 ```
   # File $DAKTOA_SRC/packages/external/demo_tpl/CMakeLists.txt
   cmake_minimum_required(VERSION 2.8)
   project("DEMO_TPL" CXX)
   SUBDIRS(src)
  ```
 In the src subdirectory of demo_tpl would be another _CMakeLists.txt_ file which essentially
 identifies the relevant source code to be compiled into a library along with defining the 
 library which Daktoa will later include, eg

 ```
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
  ```

 Note that it is possible to use Cmake's glob feature to bring in all
 source and header files, but care must be taken to avoid introducing
 `main(...)` symbols which will collide with Dakota's `main` at link
 time.


 At this point, Dakota's _CMakeLists.txt_ files will need to be
 modified to include the _Demo_ tpl library.  The following modified
 can be used to bring in the _Demo_ TPL conditioned on having `-D
 HAVE_DEMO_TPL:BOOL=ON` defined when invoking cmake to configure Dakota:

 ```
   # File $DAKTOA_SRC/packages/CMakeLists.txt

   <... snip ...>
     option(HAVE_DEMO_TPL "Build the Demo_TPL package." OFF)
   <... end snip ...>
   
   <... snip ...>
     if(HAVE_DEMO_TPL)
       add_subdirectory(external/demo_tpl)
     endif(HAVE_DEMO_TPL)
   <... end snip ...>
   
  ```

 This next modification to Dakota will allow the _Demo_ TPL to be used
 by other Dakota source code by including the necessary include paths,
 link-time libraries and needed #defines:

 ```
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
 ```

 

# Modifying Dakota to use the _Demo_  TPL

 Before making concrete changes, it is often helpful to create a simple
 Dakota test which will serve to guide the process.  This is akin to
 test-driven development which essentially creates a test which fails
 until everything has been implemented to allow it to run and pass. An
 candidate test for the current activity could be the following:

 ```
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
   ```
 For this test to run, we will need to be able to pass parsed options
 to the _Demo_ TPL and exchange parameters and response values between
 Dakota and _Demo_ TPL.  These details are presented in the following
 two sections.

## Passing Options

 Dakota maintains a master list of hierarchical options in its
 $DAKTOA_SRC/src/dakota.xml file.  Several common options associated with
 optimizers are already supported and then only need to be exposed within
 the correct hierarchy (scope).  Initially, however, it is often best to
 simply pass a native options file directly to a TPL.  Dakaota supprts
 this approach via the `options_file` specification as shown in the test
 input file.  To both expose the `demo_tpl` optimizer and associate the
 `options_file` specification with it, the dakota.xml file would be
 modified as follows:

 ```
   # File $DAKTOA_SRC/src/dakota.xml

   ... snip ...
    <!-- **** TOPLEVEL *** -->
    <keyword id="method" name="method" minOccurs="1" maxOccurs="unbounded" code="{N_mdm3(start,0,stop)}" label="Method" >

     ... snip ...
      <!-- Primary method selection alternation -->
      <oneOf label="Method (Iterative Algorithm)">

       ... snip ...

        <keyword  id="demo_tpl" name="demo_tpl" code="{N_mdm(utype,methodName_DEMO_TPL)}" label="demo_tpl" help="" minOccurs="1" group="Optimization: Local" >
          <keyword  id="options_file" name="options_file" code="{N_mdm(str,advancedOptionsFilename)}" label="Advanced Options File"  minOccurs="0" default="no advanced options file" >
            <param type="INPUT_FILE" />
          </keyword>
        </keyword>

       ... end snip ...
     ... end snip ...
   ... end snip ...
 ```

 Dakota's current parser system next needs to connect this change to
 it's internal options database and to its list of methods.  This is
 accomplished by modifying a few files as follows, eg

 ```
   # File $DAKTOA_SRC/src/NIDRProblemDescDB.cpp
   <... snip ...>
     MP2s(methodName,ROL),      // existing method
     MP2s(methodName,DEMO_TPL), // -----  our new demo_tpl method -----
     MP2s(methodName,NL2SOL),   // existing method
   <... end snip ...>
 ```

 ```
   # File $DAKTOA_SRC/src/DataMethod.hpp
   <... snip ...>
       GENIE_OPT_DARTS, GENIE_DIRECT,
       // Place Demo Opt TPL here based on current state of non-gradient flavor
       DEMO_TPL,                // -----  our new demo_tpl method -----
       // Gradient-based Optimizers / Minimizers:
       NONLINEAR_CG, OPTPP_CG, OPTPP_Q_NEWTON, OPTPP_FD_NEWTON, OPTPP_NEWTON,
   <... end snip ...>
 ```


 ```
   # File $DAKTOA_SRC/src/DataIterator.cpp
   <... snip ...>
        #ifdef HAVE_DEMO_TPL
        #include "DemoOptimizer.hpp"
        #endif
   <... end snip ...>

        Iterator* Iterator::get_iterator(ProblemDescDB& problem_db, Model& model)
        {
          unsigned short method_name = problem_db.get_ushort("method.algorithm");
   <... snip ...>
        #ifdef HAVE_DEMO_TPL
            case DEMO_TPL:      // -----  our new demo_tpl method -----
              return new DemoTPLOptimizer(problem_db, model); break;
        #endif
            default:
              switch (method_name) {
   <... end snip ...>


        /// bimap between method enums and strings; only used in this
        /// compilation unit
        static UShortStrBimap method_map =
          boost::assign::list_of<UShortStrBimap::relation>
          (HYBRID,                          "hybrid")
   <... snip ...>
          (DEMO_TPL,                        "demo_tpl")
          ;
   <... end snip ...>
 ```

 The next time Dakota is configured with the option `-D ENABLE_SPEC_MAINT:BOOL=ON`
 defined Dakota will automatically generate a file, $DAKTOA_SRC/src/dakota.input.nspec,
 based on the dakota.xml file.

 Once Dakota has been compiled with these changes, the simple test input
 file should parse and attempt to call the DemoTPLOptimizer::core_run()
 method to perform the optimization of the Dakota "text_book" example
 problem.


## Exchanging Parameters and Reponses

 Like any TPL, the _Demo_ TPL will need to exchange parameter and
 obective function values with Dakota.  For purposes of demonstration,
 an example interface between Dakota and the _Demo_ TPL can be seen
 in $DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.hpp (with
 corresponding .cpp in the same directory).  Within these files is 
 a key callback interface used by the _Demo_ TPL to obtain objective function
 values for given parater values (3 in the test above), eg:

 ```
   # File $DAKTOA_SRC/packages/external/dakota_src/DemoOptimizer.cpp

    Real
    DemoTPLOptimizer::compute_obj(const std::vector<double> & x, bool verbose)
    {
      set_variables<std::vector<double> >(x, iteratedModel, iteratedModel.current_variables());

      iteratedModel.evaluate();// default active s
      const BoolDeque& max_sense = iteratedModel.primary_response_fn_sense();

      double f = (!max_sense.empty() && max_sense[0]) ?
                 -iteratedModel.current_response().function_value(0) :
                  iteratedModel.current_response().function_value(0);

      return f;
    }
 ```

 In this instance, the _Demo_ TPL uses `std::vector<double>` as its native
 parameter vector data type and is calling back to the example problem
 via an interface to Dakota to obtain a single `double` (aliased to `Real`
 in Dakota) obective function value for a given set of parameter values.
 These data exchanges are facilitated by used of "data adapters" supplied
 by Dakota with the `set_variables<>(...)` helper utilized in this case.


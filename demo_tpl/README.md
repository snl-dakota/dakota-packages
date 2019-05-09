
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

 

# Modifying Daktoa to use the _Demo_  TPL


## Passing Options

## Exchanging Paramters and Reponses



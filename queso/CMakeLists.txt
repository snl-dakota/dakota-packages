# Experimental QUESO CMake that builds only the sources (no tests,
# examples, etc.).  Does not properly manage all QUESO
# build/conditional compile settings.

cmake_minimum_required(VERSION 2.8.9)
project(Queso CXX)

# -----
# Required packages
# -----

# MPI is optional and Dakota manages MPI for now
#find_package(MPI REQUIRED)

# This protection shouldn't be necessary, but having trouble with Acro
# and non-system-provided Boost.  Probably need to change all
# invocations to Boost_FOUND instead of Boost_DIR.
if(NOT Boost_FOUND)
  find_package(Boost 1.44 REQUIRED COMPONENTS program_options)
endif()

find_package(GSL REQUIRED) # Version 1.15 or newer?
# Required by queso.h.in (cannot prefix with QUESO_)
set(GSL_LIBS ${GSL_LIBRARY_DIRS})


# -----
# CMake defines for generated headers
# -----

if(DAKOTA_HAVE_MPI)
  set(QUESO_HAVE_MPI 1)
endif()

# Boost options
# One CMake option QUESO_HAVE_BOOST to control all the following
#  QUESO_HAVE_BOOST
#  QUESO_HAVE_BOOST_MATH_SPECIAL_FUNCTIONS_HPP
#  QUESO_HAVE_BOOST_PROGRAM_OPTIONS_HPP
#  QUESO_HAVE_BOOST_SCOPED_PTR_HPP
#  QUESO_HAVE_BOOST_SHARED_PTR_HPP
set(QUESO_HAVE_BOOST 1)


# TODO: Check for C++11 support once using CMake 3.x (CXX_STANDARD)


# Check for include files
include(CheckIncludeFile)
check_include_file("dlfcn.h" QUESO_HAVE_DLFCN_H)
check_include_file("stdint.h" QUESO_HAVE_STDINT_H)

# TODO: Add other include files from config_queso.h.cmake.in


# -----
# Build Information
# Definitions used by queso.h.in and config_queso.h.cmake.in
# -----

# TODO: see if we can get upstream to consolidate the two configured
# header files

set(GENERIC_MAJOR_VERSION 0)
set(GENERIC_MINOR_VERSION 56)
set(GENERIC_MICRO_VERSION 0)

set(BUILD_USER "anonymous")
set(BUILD_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
#set(BUILD_HOST "${CMAKE_SYSTEM_NAME}")
set(BUILD_HOST "unknown")
# this deviates from QUEOS, which uses a static build date in queso.h,
# but dynamic in config_queso.h; need to embed the quotes for macro
# expansion
# TODO: embed date when using CMake 2.8.12 or newer
#set(BUILD_DATE "YYYY-MM-DD HH:MM")
set(BUILD_DATE "\"__DATE__ \" \"__TIME__\"")
# TODO: short SHA1 from upstream queso git repo:
set(BUILD_VERSION "N/A")
set(VERSION 
  "${GENERIC_MAJOR_VERSION}.${GENERIC_MINOR_VERSION}.${GENERIC_MICRO_VERSION}")
set(BUILD_DEVSTATUS "Development Build")
set(CXX "${CMAKE_CXX_COMPILER}")
# TODO: Following should depend on build type
set(CXXFLAGS "${CMAKE_CXX_FLAGS}")


# -----
# Generate include files
# -----

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/queso)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/core/inc/queso.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/queso/queso.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_queso.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/queso/config_queso.h)


# -----
# Properties for compilation/link
# -----

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include 
  ${MPI_CXX_INCLUDE_PATH}
  ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIRS}
  ${GSL_INCLUDE_DIRS}
  )

# Don't yet need since not linking anything here...
#link_directories(${Boost_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS})

# Don't force, so parent can override
set(BUILD_SHARED_LIBS TRUE CACHE BOOL "QUESO building shared libraries")


# -----
# Create libraries and copy headers
# -----

set(queso_sources)
foreach(queso_subdir basic core misc stats gp surrogates)

  # Mimic autotools build process and put headers in build tree
  # Likely can eliminate...
  file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/src/${queso_subdir}/inc/*.h)
  foreach(h ${headers})
    configure_file(${h} ${CMAKE_CURRENT_BINARY_DIR}/include/queso/ COPYONLY)
  endforeach()

  file(GLOB dir_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    src/${queso_subdir}/src/*.C)
  list(APPEND queso_sources ${dir_sources})

  # Add one library per directory
  #add_library(queso_${queso_subdir} ${dir_sources})
  #target_link_libraries(queso_${queso_subdir} ${Boost_LIBRARIES})

endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/getpot/getpot.h 
  ${CMAKE_CURRENT_BINARY_DIR}/include/queso/ COPYONLY)

# Add one library in total
add_library(queso ${queso_sources})
target_link_libraries(queso 
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})
# Apply MPI compiler/link options from Dakota MPI
DakotaApplyMPISettings(queso)

install(TARGETS queso EXPORT ${ExportTarget} DESTINATION lib)

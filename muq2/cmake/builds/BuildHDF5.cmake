include(ExternalProject)

if(NOT DEFINED MUQ_INTERNAL_HDF5_VERSION)
  set(MUQ_INTERNAL_HDF5_VERSION "1.8.19")
endif()

if(NOT HDF5_EXTERNAL_SOURCE)

  set(HDF5_EXTERNAL_SOURCE https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-${MUQ_INTERNAL_HDF5_VERSION}/src/CMake-hdf5-${MUQ_INTERNAL_HDF5_VERSION}.tar.gz)
  message(STATUS "Will download HDF5 from ${HDF5_EXTERNAL_SOURCE} during compile.")

endif()

if(MUQ_USE_OPENMPI)
	set(HDF5_PARALLEL_FLAG	"--enable-parallel" "CFLAGS=-fPIC -I${MPI_INCLUDE_DIR}")
endif()

set(HDF5_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/muq_external/)
ExternalProject_Add(
  HDF5
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/hdf5
    URL ${HDF5_EXTERNAL_SOURCE}
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}; ${CMAKE_CURRENT_BINARY_DIR}/external/hdf5/src/HDF5/hdf5-${MUQ_INTERNAL_HDF5_VERSION}/configure  CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ${HDF5_PARALLEL_FLAG} --prefix=${HDF5_INSTALL_DIR} --enable-production --disable-examples
    BUILD_COMMAND make install
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)



set_property( TARGET HDF5 PROPERTY FOLDER "Externals")


  set(HDF5_INCLUDE_DIRS "${HDF5_INSTALL_DIR}include" )
  if(MUQ_USE_OPENMPI)
      set( HDF5_LIBRARIES ${HDF5_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5_hl${CMAKE_STATIC_LIBRARY_SUFFIX} ${HDF5_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
      set( HDF5_LIBRARIES ${HDF5_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}hdf5_hl${CMAKE_SHARED_LIBRARY_SUFFIX} ${HDF5_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}hdf5${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()

  set( HDF5_LIBRARIES_STATIC ${HDF5_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5_hl${CMAKE_STATIC_LIBRARY_SUFFIX} ${HDF5_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5${CMAKE_STATIC_LIBRARY_SUFFIX})

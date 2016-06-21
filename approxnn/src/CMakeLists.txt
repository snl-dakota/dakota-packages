# Simple CMakeLists.txt for ANN/src only added for Dakota project

cmake_minimum_required(VERSION 2.8)

project(ApproxNN CXX)

set(ExportTarget ${CMAKE_PROJECT_NAME}-targets CACHE STRING 
  "Name for the export target for ${CMAKE_PROJECT_NAME}")

set(ApproxNN_SOURCES 
  ANN.cpp brute.cpp kd_tree.cpp kd_util.cpp kd_split.cpp
  kd_dump.cpp kd_search.cpp kd_pr_search.cpp kd_fix_rad_search.cpp
  bd_tree.cpp bd_search.cpp bd_pr_search.cpp bd_fix_rad_search.cpp
  perf.cpp
  )

set(ApproxNN_HEADERS
  kd_tree.h kd_split.h kd_util.h kd_search.h
  kd_pr_search.h kd_fix_rad_search.h pr_queue.h pr_queue_k.h
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_library(approxnn ${ApproxNN_SOURCES})

install(TARGETS approxnn EXPORT ${ExportTarget} DESTINATION lib)
install(FILES ${ApproxNN_HEADERS} DESTINATION include)

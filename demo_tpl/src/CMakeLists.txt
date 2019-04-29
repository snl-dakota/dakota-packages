## DEMO_TPL library sources.
set(demo_tpl_HEADERS
  execute.hpp
  initialize.hpp
)

set(demo_tpl_SOURCES
  execute.cpp
  initialize.cpp
)

## Set the DEMO_TPL library name.
add_library(demo_tpl ${demo_tpl_SOURCES})

install(TARGETS demo_tpl EXPORT ${ExportTarget} DESTINATION lib)

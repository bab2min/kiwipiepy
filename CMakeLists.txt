project(Kiwipiepy)

add_subdirectory( Kiwi )
include_directories(${INCLUDE_DIRS}
  Kiwi/include
)
link_directories(${LIBRARY_DIRS})

add_library("_kiwipiepy" SHARED
  src/KiwiPy.cpp
)
target_link_libraries("_kiwipiepy" PRIVATE kiwi_static)

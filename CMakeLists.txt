cmake_minimum_required(VERSION 3.9)

project(Kiwipiepy)

set ( CMAKE_CXX_STANDARD 11 )
set ( CMAKE_VERBOSE_MAKEFILE true )

add_subdirectory( Kiwi )
include_directories(${INCLUDE_DIRS}
  Kiwi/include
)
link_directories(${LIBRARY_DIRS})

add_library("_kiwipiepy" SHARED
  src/KiwiPy.cpp
)
target_link_libraries("_kiwipiepy" PRIVATE 
  kiwi_static
  ${LIBRARIES}
)

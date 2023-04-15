cmake_minimum_required(VERSION 3.9)

project(Kiwipiepy)

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_VERBOSE_MAKEFILE true )

set ( CMAKE_SKIP_RPATH ON )
set ( CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version" )

add_subdirectory( Kiwi )

if(KIWI_USE_MIMALLOC)
  message(STATUS "Use mimalloc allocators")
  set ( ADDITIONAL_FLAGS "-DKIWI_USE_MIMALLOC" )
  include_directories( Kiwi/third_party/mimalloc/include )
endif()

if(MSVC)
  set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -Zc:__cplusplus -Zi -Od ${ADDITIONAL_FLAGS}" )
  set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )

  set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -Zc:__cplusplus -O2 -Oi -Gy -DKIWI_USE_BTREE ${ADDITIONAL_FLAGS}" )
  set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )

  set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -Zi /bigobj")
  set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
else()
  set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -g3 -O0 ${ADDITIONAL_FLAGS}" )
  set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )
  set ( CMAKE_EXE_LINKER_FLAGS_DEBUG "-DDEBUG -DLINKER_FLAGS" )

  set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O3 -DKIWI_USE_BTREE ${ADDITIONAL_FLAGS}" )
  set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )
  set ( CMAKE_EXE_LINKER_FLAGS_RELEASE "-DRELEASE -DLINKER_FLAGS" )

  set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g3")
  set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
endif()

if (APPLE)
  message(STATUS "Apple Clang: use undefined=dynamic_lookup")
  set(CMAKE_OSX_ARCHITECTURES "${KIWI_CPU_ARCH}")
  set ( CMAKE_C_FLAGS_DEBUG "-undefined dynamic_lookup ${CMAKE_C_FLAGS_DEBUG}" )
  set ( CMAKE_CXX_FLAGS_DEBUG "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS_DEBUG}" )
  set ( CMAKE_C_FLAGS_RELEASE "-undefined dynamic_lookup ${CMAKE_C_FLAGS_RELEASE}" )
  set ( CMAKE_CXX_FLAGS_RELEASE "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS_RELEASE}" )
  set ( CMAKE_C_FLAGS_RELWITHDEBINFO "-undefined dynamic_lookup ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

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


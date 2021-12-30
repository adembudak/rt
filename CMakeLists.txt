cmake_minimum_required(VERSION 3.18)

project(raytracer
    VERSION 1.0.0.0
    DESCRIPTION "A simple raytracer based on the book \"Computer Graphics from Scratch (https://gabrielgambetta.com/computer-graphics-from-scratch/)\""
    HOMEPAGE_URL https://github.com/adembudak/raytracer
    LANGUAGES CXX)
include(FetchContent)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES CACHE BOOL "" FORCE)
set(CMAKE_CXX_EXTENSIONS NO CACHE BOOL "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/stb
	GIT_PROGRESS OFF
	GIT_SHALLOW ON)

FetchContent_MakeAvailable(stb)

add_library(stb_image_write INTERFACE)
target_sources(stb_image_write INTERFACE ${PROJECT_SOURCE_DIR}/3rdparty/stb/stb_image_write.h)
target_include_directories(stb_image_write INTERFACE ${PROJECT_SOURCE_DIR}/3rdparty/stb)

add_executable(raytracer main.cpp)
target_link_libraries(raytracer PUBLIC stb_image_write)
target_include_directories(raytracer PUBLIC ${PROJECT_SOURCE_DIR}/3rdparty 
	${PROJECT_SOURCE_DIR}/include)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
   target_compile_options(raytracer PUBLIC -Wall -Wextra -Wconversion)
endif()


cmake_minimum_required(VERSION 3.16)
project(zenl_sdk_smoke LANGUAGES CXX)

find_package(ZENL REQUIRED)

add_executable(zenl_sdk_smoke main.cpp)
target_compile_features(zenl_sdk_smoke PRIVATE cxx_std_17)
target_link_libraries(zenl_sdk_smoke PRIVATE ZENL::zenl)

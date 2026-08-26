# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-src"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-build"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/tmp"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/src"
  "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_det/_deps/stb-subbuild/stb-populate-prefix/src/stb-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

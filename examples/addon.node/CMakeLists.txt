cmake_minimum_required(VERSION 3.14)

# macOS deployment target - set before project() to ensure it applies to all targets
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Minimum macOS version")
endif()

project(whisper-addon)

set(TARGET addon.node)

# ============================================================================
# Base settings for cmake-js
# ============================================================================
add_definitions(-DNAPI_VERSION=4)
include_directories(${CMAKE_JS_INC})

# ============================================================================
# Include N-API wrappers
# ============================================================================
execute_process(COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
)
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# ============================================================================
# Build Mode: Standalone (self-contained) vs Integrated (with parent project)
# ============================================================================

# Check if we're being built as part of the main whisper.cpp project
if(TARGET whisper AND TARGET common)
    message(STATUS "Building addon.node in integrated mode (using existing whisper/common targets)")
    set(ADDON_STANDALONE OFF)
else()
    message(STATUS "Building addon.node in standalone mode (self-contained)")
    set(ADDON_STANDALONE ON)
endif()

if(ADDON_STANDALONE)
    # ========================================================================
    # Standalone Mode: Build everything from source
    # ========================================================================

    # Force static libraries for self-contained build
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

    # Set paths relative to whisper.cpp root
    get_filename_component(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

    message(STATUS "Whisper root: ${WHISPER_ROOT}")

    # Add ggml subdirectory
    if(NOT TARGET ggml)
        add_subdirectory(${WHISPER_ROOT}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
    endif()

    # ========================================================================
    # Build whisper library from source
    # ========================================================================
    set(WHISPER_SOURCES
        ${WHISPER_ROOT}/src/whisper.cpp
    )

    # Extract version from main CMakeLists.txt
    file(READ "${WHISPER_ROOT}/CMakeLists.txt" WHISPER_CMAKE_CONTENT)
    string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" _ "${WHISPER_CMAKE_CONTENT}")
    if(CMAKE_MATCH_1)
        set(WHISPER_VERSION "${CMAKE_MATCH_1}")
    else()
        set(WHISPER_VERSION "unknown")
    endif()
    message(STATUS "Whisper version: ${WHISPER_VERSION}")

    add_library(whisper_static STATIC ${WHISPER_SOURCES})
    target_include_directories(whisper_static PUBLIC
        ${WHISPER_ROOT}/src
        ${WHISPER_ROOT}/include
    )
    target_compile_features(whisper_static PUBLIC cxx_std_11)
    target_compile_definitions(whisper_static PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")
    target_link_libraries(whisper_static PUBLIC ggml)

    find_package(Threads REQUIRED)
    target_link_libraries(whisper_static PUBLIC Threads::Threads)

    # ========================================================================
    # Build common library from source
    # ========================================================================
    set(COMMON_SOURCES
        ${WHISPER_ROOT}/examples/common.cpp
        ${WHISPER_ROOT}/examples/common-ggml.cpp
        ${WHISPER_ROOT}/examples/common-whisper.cpp
        ${WHISPER_ROOT}/examples/grammar-parser.cpp
    )

    add_library(common_static STATIC ${COMMON_SOURCES})
    target_include_directories(common_static PUBLIC
        ${WHISPER_ROOT}/examples
        ${WHISPER_ROOT}/include
    )
    target_link_libraries(common_static PRIVATE whisper_static)
    set_target_properties(common_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set(WHISPER_LIB whisper_static)
    set(COMMON_LIB common_static)
else()
    # ========================================================================
    # Integrated Mode: Use existing targets from parent project
    # ========================================================================
    include(DefaultTargetOptions)

    set(WHISPER_LIB whisper)
    set(COMMON_LIB common)
endif()

# ============================================================================
# Build the addon
# ============================================================================
add_library(${TARGET} SHARED ${CMAKE_JS_SRC} addon.cpp)
set_target_properties(${TARGET} PROPERTIES PREFIX "" SUFFIX ".node")

target_include_directories(${TARGET} PRIVATE ${NODE_ADDON_API_DIR})

if(ADDON_STANDALONE)
    # Include paths for standalone build
    get_filename_component(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    target_include_directories(${TARGET} PRIVATE
        ${WHISPER_ROOT}/examples
        ${WHISPER_ROOT}/include
        ${WHISPER_ROOT}/src
    )
endif()

target_link_libraries(${TARGET}
    ${CMAKE_JS_LIB}
    ${COMMON_LIB}
    ${WHISPER_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
)

# ============================================================================
# MSVC specific settings
# ============================================================================
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

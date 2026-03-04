cmake_minimum_required(VERSION 3.14)

# macOS deployment target - set before project() to ensure it applies to all targets
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Minimum macOS version")
endif()

project(whisper-addon)

set(TARGET addon.node)

# ============================================================================
# Build options for GPU backends
# ============================================================================
# Auto-enable Vulkan on Windows by default
if(WIN32)
    set(ADDON_VULKAN_DEFAULT ON)
    set(ADDON_OPENVINO_DEFAULT ON)
else()
    set(ADDON_VULKAN_DEFAULT OFF)
    set(ADDON_OPENVINO_DEFAULT OFF)
endif()

option(ADDON_VULKAN   "Enable Vulkan GPU support (Windows/Linux)"  ${ADDON_VULKAN_DEFAULT})
option(ADDON_OPENVINO "Enable OpenVINO encoder support (Intel)"    ${ADDON_OPENVINO_DEFAULT})

# ============================================================================
# MSVC static runtime - eliminates VC++ Redistributable requirement for users
# Must be set before any targets (add_library/add_subdirectory)
# ============================================================================
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

# ============================================================================
# MSVC UTF-8 encoding fix (must be set early, before adding subdirectories)
# whisper.cpp contains Unicode characters that MSVC can't handle without this
# ============================================================================
if(MSVC)
    add_compile_options(/utf-8)
endif()

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

    # ========================================================================
    # Configure ggml GPU backends before adding subdirectory
    # ========================================================================
    if(ADDON_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
        message(STATUS "Vulkan GPU support: ENABLED")
    endif()

    # Add ggml subdirectory
    if(NOT TARGET ggml)
        add_subdirectory(${WHISPER_ROOT}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
    endif()

    # ========================================================================
    # Core ML support (macOS only) - always enabled for runtime switching
    # ========================================================================
    if(APPLE)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        find_library(COREML_FRAMEWORK CoreML)

        if(COREML_FRAMEWORK)
            message(STATUS "CoreML framework found - enabling Core ML support")
            set(WHISPER_COREML_ENABLED ON)

            # Build whisper.coreml static library
            set(COREML_SOURCES
                ${WHISPER_ROOT}/src/coreml/whisper-compat.m
                ${WHISPER_ROOT}/src/coreml/whisper-encoder.mm
                ${WHISPER_ROOT}/src/coreml/whisper-encoder-impl.m
            )

            add_library(whisper_coreml STATIC ${COREML_SOURCES})
            target_include_directories(whisper_coreml PUBLIC
                ${WHISPER_ROOT}/src
                ${WHISPER_ROOT}/src/coreml
            )
            target_link_libraries(whisper_coreml PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})
            set_target_properties(whisper_coreml PROPERTIES
                COMPILE_FLAGS "-fobjc-arc"
                XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
            )
        else()
            message(STATUS "CoreML framework not found - Core ML support disabled")
            set(WHISPER_COREML_ENABLED OFF)
        endif()
    else()
        set(WHISPER_COREML_ENABLED OFF)
    endif()

    # ========================================================================
    # OpenVINO support (Intel CPUs/GPUs) - auto-enabled on Windows if found
    # ========================================================================
    set(WHISPER_OPENVINO_ENABLED OFF)
    
    if(ADDON_OPENVINO)
        find_package(OpenVINO COMPONENTS Runtime QUIET)
        if(OpenVINO_FOUND)
            message(STATUS "OpenVINO found - enabling OpenVINO encoder support")
            set(WHISPER_OPENVINO_ENABLED ON)

            # Build whisper.openvino static library
            set(OPENVINO_SOURCES
                ${WHISPER_ROOT}/src/openvino/whisper-openvino-encoder.cpp
            )

            add_library(whisper_openvino STATIC ${OPENVINO_SOURCES})
            target_include_directories(whisper_openvino PUBLIC
                ${WHISPER_ROOT}/src
                ${WHISPER_ROOT}/src/openvino
                ${WHISPER_ROOT}/ggml/include
            )
            target_link_libraries(whisper_openvino PRIVATE openvino::runtime)
            set_target_properties(whisper_openvino PROPERTIES POSITION_INDEPENDENT_CODE ON)
        else()
            if(WIN32)
                message(STATUS "OpenVINO not found - OpenVINO support disabled (install OpenVINO for Intel acceleration)")
            else()
                message(WARNING "OpenVINO requested but not found - OpenVINO support disabled")
            endif()
        endif()
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

    # Add Core ML support to whisper if available
    if(WHISPER_COREML_ENABLED)
        target_compile_definitions(whisper_static PRIVATE WHISPER_USE_COREML)
        target_compile_definitions(whisper_static PRIVATE WHISPER_COREML_ALLOW_FALLBACK)
        target_link_libraries(whisper_static PRIVATE whisper_coreml)
    endif()

    # Add OpenVINO support to whisper if available
    if(WHISPER_OPENVINO_ENABLED)
        target_compile_definitions(whisper_static PRIVATE WHISPER_USE_OPENVINO)
        target_link_libraries(whisper_static PRIVATE whisper_openvino)
    endif()

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

    # Pass OpenVINO enabled flag to addon
    if(WHISPER_OPENVINO_ENABLED)
        target_compile_definitions(${TARGET} PRIVATE WHISPER_USE_OPENVINO)
    endif()
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

# ============================================================================
# Delay-load Vulkan DLL (Windows) - allows graceful CPU fallback
# Without this, the addon fails to load entirely if vulkan-1.dll is missing.
# With /DELAYLOAD, vulkan-1.dll is only loaded when the first Vulkan function
# is called. If GPU init fails before that (or falls back to CPU), the DLL
# is never needed.
# ============================================================================
if(MSVC AND ADDON_VULKAN)
    target_link_options(${TARGET} PRIVATE "/DELAYLOAD:vulkan-1.dll")
    target_link_libraries(${TARGET} delayimp.lib)
endif()

# ============================================================================
# Delay-load OpenVINO DLLs (Windows) - allows graceful fallback on non-Intel
# Without this, the addon fails to load entirely if OpenVINO DLLs are missing
# (e.g. inside Electron .asar archives, or on AMD/non-Intel systems).
# With /DELAYLOAD, the DLLs are only loaded when OpenVINO is actually used.
# ============================================================================
if(MSVC AND WHISPER_OPENVINO_ENABLED)
    target_link_options(${TARGET} PRIVATE
        "/DELAYLOAD:openvino.dll"
        "/DELAYLOAD:openvino_ir_frontend.dll"
        "/DELAYLOAD:tbb12.dll"
        "/DELAYLOAD:pugixml.dll"
    )
    # delayimp.lib may already be linked (from Vulkan above), but CMake deduplicates
    target_link_libraries(${TARGET} delayimp.lib)
endif()

# ============================================================================
# Copy runtime DLLs to output directory (Windows)
# ============================================================================
if(WIN32 AND WHISPER_OPENVINO_ENABLED)
    # Find OpenVINO runtime DLLs
    get_target_property(OPENVINO_RUNTIME_LIB openvino::runtime IMPORTED_LOCATION_RELEASE)
    if(NOT OPENVINO_RUNTIME_LIB)
        get_target_property(OPENVINO_RUNTIME_LIB openvino::runtime IMPORTED_LOCATION)
    endif()

    if(OPENVINO_RUNTIME_LIB)
        get_filename_component(OPENVINO_LIB_DIR "${OPENVINO_RUNTIME_LIB}" DIRECTORY)
        message(STATUS "OpenVINO library directory: ${OPENVINO_LIB_DIR}")

        # Core DLLs (in main bin directory)
        set(OPENVINO_CORE_DLLS
            openvino.dll
            openvino_ir_frontend.dll
            tbb12.dll
            pugixml.dll
        )

        # Plugin DLLs (may be in versioned subdirectory for conda installs)
        set(OPENVINO_PLUGIN_DLLS
            openvino_intel_cpu_plugin.dll
            openvino_intel_gpu_plugin.dll
            openvino_auto_plugin.dll
        )

        # Copy core DLLs from main directory
        foreach(dll ${OPENVINO_CORE_DLLS})
            set(dll_path "${OPENVINO_LIB_DIR}/${dll}")
            if(EXISTS "${dll_path}")
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_path}"
                        "$<TARGET_FILE_DIR:${TARGET}>"
                    COMMENT "Copying ${dll}"
                )
            else()
                message(STATUS "OpenVINO DLL not found (skipping): ${dll_path}")
            endif()
        endforeach()

        # Find plugins - check main dir first, then versioned subdirectories (conda layout)
        foreach(dll ${OPENVINO_PLUGIN_DLLS})
            set(dll_found FALSE)
            set(dll_path "${OPENVINO_LIB_DIR}/${dll}")

            if(EXISTS "${dll_path}")
                set(dll_found TRUE)
            else()
                # Search in versioned subdirectories (e.g., openvino-2024.6.0/)
                file(GLOB OPENVINO_SUBDIRS "${OPENVINO_LIB_DIR}/openvino-*")
                foreach(subdir ${OPENVINO_SUBDIRS})
                    if(IS_DIRECTORY "${subdir}")
                        set(subdir_dll_path "${subdir}/${dll}")
                        if(EXISTS "${subdir_dll_path}")
                            set(dll_path "${subdir_dll_path}")
                            set(dll_found TRUE)
                            break()
                        endif()
                    endif()
                endforeach()
            endif()

            if(dll_found)
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_path}"
                        "$<TARGET_FILE_DIR:${TARGET}>"
                    COMMENT "Copying ${dll}"
                )
            else()
                message(STATUS "OpenVINO plugin not found (skipping): ${dll}")
            endif()
        endforeach()
    endif()
endif()

# ============================================================================
# Copy MSVC runtime DLLs to output directory (Windows)
# Bundled OpenVINO/TBB DLLs are built with /MD and require these at runtime.
# Without bundling them, users need to install VC++ Redistributable separately.
# ============================================================================
if(MSVC)
    # Find MSVC redist directory from compiler path
    get_filename_component(MSVC_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    # Compiler is at: .../MSVC/<ver>/bin/Host<arch>/<arch>/cl.exe
    # Redist is at:   .../MSVC/<ver>/Redist/MSVC/<redist_ver>/<arch>/Microsoft.VC143.CRT/
    get_filename_component(MSVC_TOOLS_DIR "${MSVC_COMPILER_DIR}/../../.." ABSOLUTE)

    # Determine target architecture for redist path
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MSVC_REDIST_ARCH "x64")
    else()
        set(MSVC_REDIST_ARCH "x86")
    endif()

    # Search for CRT DLLs (msvcp140, vcruntime140, vcruntime140_1)
    set(MSVC_CRT_DLLS
        msvcp140.dll
        vcruntime140.dll
        vcruntime140_1.dll
    )

    file(GLOB MSVC_REDIST_DIRS "${MSVC_TOOLS_DIR}/Redist/MSVC/*/")
    foreach(redist_dir ${MSVC_REDIST_DIRS})
        set(MSVC_CRT_DIR "${redist_dir}${MSVC_REDIST_ARCH}/Microsoft.VC143.CRT")
        if(IS_DIRECTORY "${MSVC_CRT_DIR}")
            message(STATUS "MSVC CRT directory: ${MSVC_CRT_DIR}")
            foreach(dll ${MSVC_CRT_DLLS})
                set(dll_path "${MSVC_CRT_DIR}/${dll}")
                if(EXISTS "${dll_path}")
                    add_custom_command(TARGET ${TARGET} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${dll_path}"
                            "$<TARGET_FILE_DIR:${TARGET}>"
                        COMMENT "Copying MSVC runtime: ${dll}"
                    )
                endif()
            endforeach()
            break()
        endif()
    endforeach()

    # Search for OpenMP runtime DLL (vcomp140)
    foreach(redist_dir ${MSVC_REDIST_DIRS})
        set(MSVC_OMP_DIR "${redist_dir}${MSVC_REDIST_ARCH}/Microsoft.VC143.OpenMP")
        if(IS_DIRECTORY "${MSVC_OMP_DIR}")
            set(vcomp_path "${MSVC_OMP_DIR}/vcomp140.dll")
            if(EXISTS "${vcomp_path}")
                message(STATUS "MSVC OpenMP directory: ${MSVC_OMP_DIR}")
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${vcomp_path}"
                        "$<TARGET_FILE_DIR:${TARGET}>"
                    COMMENT "Copying MSVC runtime: vcomp140.dll"
                )
            endif()
            break()
        endif()
    endforeach()
endif()

cmake_minimum_required(VERSION 3.20)

project(gemma4_mlx LANGUAGES CXX)

option(GEMMA4D_REQUIRE_MLX "Require MLX during CMake configuration." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gemma4_mlx STATIC
    src/model_manifest.cc
    src/native_model.cc
    src/runtime.cc
)

target_include_directories(gemma4_mlx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(gemma4_mlx
    PRIVATE
        GEMMA4D_MLX_LM_HELPER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/gemma4d_mlx_lm_helper.py"
)

if(GEMMA4D_REQUIRE_MLX)
    find_package(MLX CONFIG REQUIRED)
    target_compile_definitions(gemma4_mlx PRIVATE GEMMA4D_MLX_AVAILABLE=1)
    if(TARGET mlx)
        target_link_libraries(gemma4_mlx PRIVATE mlx)
    elseif(TARGET MLX::mlx)
        target_link_libraries(gemma4_mlx PRIVATE MLX::mlx)
    else()
        target_include_directories(gemma4_mlx PRIVATE ${MLX_INCLUDE_DIRS})
        target_link_libraries(gemma4_mlx PRIVATE ${MLX_LIBRARIES})
    endif()
else()
    message(STATUS "MLX is not required for the M01 FFI smoke build. Configure with -DGEMMA4D_REQUIRE_MLX=ON to require the real MLX dependency.")
endif()

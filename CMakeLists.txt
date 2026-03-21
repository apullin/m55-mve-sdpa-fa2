cmake_minimum_required(VERSION 3.20)

project(arm_mve_tiled_attention C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(ARM_M55_MVE "Enable CMSIS-DSP Helium options for Cortex-M55 builds" OFF)
option(ARM_M55_QEMU "Build a bare-metal Cortex-M55 binary for QEMU mps3-an547" OFF)
option(ARM_PROFILE_CYCLES "Enable DWT cycle profiling on Cortex-M55 builds" OFF)
option(ARM_REQUIRE_MVE "Fail Cortex-M55 builds if Helium/MVE support is not active" ON)
set(ARM_FLOAT_ABI "hard" CACHE STRING "Floating-point ABI for Cortex-M55 cross builds (softfp or hard)")

set(MODEL_DIM 24 CACHE STRING "Model feature dimension")
set(MODEL_ATTN_DIM 32 CACHE STRING "Internal attention width")
set(MODEL_FFN_DIM 48 CACHE STRING "Feed-forward hidden width")
set(MODEL_NUM_HEADS 2 CACHE STRING "Number of attention heads")
set(MODEL_NUM_LAYERS 3 CACHE STRING "Number of stacked attention layers")
set(MODEL_INPUT_SEQ_LEN 1200 CACHE STRING "Input sequence length")
set(MODEL_NUM_CLASSES 4 CACHE STRING "Number of classifier outputs")
set(MODEL_QUERY_TILE 30 CACHE STRING "Attention query tile size")
set(MODEL_KEY_TILE 150 CACHE STRING "Attention key tile size")

if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
  set(ARM_M55_QEMU ON CACHE BOOL "" FORCE)
  set(ARM_M55_MVE ON CACHE BOOL "" FORCE)
  set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
  set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>\n<CMAKE_RANLIB> <TARGET>")
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>\n<CMAKE_RANLIB> <TARGET>")
endif()

if(ARM_M55_MVE)
  set(HOST OFF CACHE BOOL "" FORCE)
  set(MVEI ON CACHE BOOL "" FORCE)
  set(MVEF OFF CACHE BOOL "" FORCE)
else()
  set(HOST ON CACHE BOOL "" FORCE)
  set(MVEI OFF CACHE BOOL "" FORCE)
  set(MVEF OFF CACHE BOOL "" FORCE)
endif()

set(LOOPUNROLL ON CACHE BOOL "" FORCE)
set(MATRIXCHECK OFF CACHE BOOL "" FORCE)

set(CMSIS_5_DIR "" CACHE PATH "Path to a CMSIS_5 checkout")

if(ARM_M55_QEMU AND NOT CMSIS_5_DIR)
  set(_cmsis5_candidates
    "${CMAKE_CURRENT_SOURCE_DIR}/../jax_to_embedded/cm4-qemu/CMSIS_5"
    "${CMAKE_CURRENT_SOURCE_DIR}/../CMSIS_5"
  )
  foreach(_candidate IN LISTS _cmsis5_candidates)
    if(EXISTS "${_candidate}/CMSIS/Core/Include/core_cm55.h")
      set(CMSIS_5_DIR "${_candidate}" CACHE PATH "" FORCE)
      break()
    endif()
  endforeach()
endif()

if(ARM_M55_QEMU)
  if(NOT ARM_FLOAT_ABI STREQUAL "softfp" AND NOT ARM_FLOAT_ABI STREQUAL "hard")
    message(FATAL_ERROR "ARM_FLOAT_ABI must be one of: softfp, hard")
  endif()
  if(NOT EXISTS "${CMSIS_5_DIR}/CMSIS/Core/Include/core_cm55.h")
    message(FATAL_ERROR "ARM_M55_QEMU requires CMSIS_5_DIR to point to a CMSIS_5 checkout with Cortex-M55 core headers")
  endif()
  set(CMSISCORE "${CMSIS_5_DIR}/CMSIS/Core" CACHE PATH "" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(MODEL_DATA_HEADER "${GENERATED_DIR}/model_data.h")

set(GEN_MODEL_ARGS
  --model-dim "${MODEL_DIM}"
  --attn-dim "${MODEL_ATTN_DIM}"
  --ffn-dim "${MODEL_FFN_DIM}"
  --num-heads "${MODEL_NUM_HEADS}"
  --num-layers "${MODEL_NUM_LAYERS}"
  --input-seq-len "${MODEL_INPUT_SEQ_LEN}"
  --num-classes "${MODEL_NUM_CLASSES}"
  --query-tile "${MODEL_QUERY_TILE}"
  --key-tile "${MODEL_KEY_TILE}"
)

add_custom_command(
  OUTPUT "${MODEL_DATA_HEADER}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${GENERATED_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_model_data.py" "${MODEL_DATA_HEADER}" ${GEN_MODEL_ARGS}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_model_data.py"
  VERBATIM
)

add_custom_target(generate_model_data DEPENDS "${MODEL_DATA_HEADER}")

if(NOT ARM_M55_QEMU)
  add_subdirectory(../CMSIS-DSP cmake-cmsis-dsp)
endif()

add_executable(
  tiled_attention
  src/main.c
  "${MODEL_DATA_HEADER}"
)

add_dependencies(tiled_attention generate_model_data)

target_include_directories(
  tiled_attention
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${GENERATED_DIR}"
)

if(ARM_M55_QEMU)
  target_sources(
    tiled_attention
    PRIVATE
    "../CMSIS-DSP/Source/MatrixFunctions/arm_mat_mult_q7.c"
    "../CMSIS-DSP/Source/MatrixFunctions/arm_mat_trans_q7.c"
    "../CMSIS-DSP/Source/BasicMathFunctions/arm_add_q7.c"
    "../CMSIS-DSP/Source/BasicMathFunctions/arm_clip_q7.c"
    "../CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_q7.c"
    "../CMSIS-DSP/Source/FastMathFunctions/arm_sqrt_q31.c"
    "../CMSIS-DSP/Source/CommonTables/arm_common_tables.c"
    "../CMSIS-DSP/Source/SupportFunctions/arm_fill_q7.c"
    "../CMSIS-DSP/Source/SupportFunctions/arm_copy_q7.c"
    "${CMSIS_5_DIR}/Device/ARM/ARMCM55/Source/startup_ARMCM55.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform/qemu-mps3-an547/system_ARMCM55.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform/qemu-mps3-an547/syscalls_semihost.c"
  )

  target_include_directories(
    tiled_attention
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../CMSIS-DSP/Include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../CMSIS-DSP/PrivateInclude"
    "${CMSIS_5_DIR}/CMSIS/Core/Include"
    "${CMSIS_5_DIR}/Device/ARM/ARMCM55/Include"
  )

  target_compile_definitions(tiled_attention PRIVATE ARMCM55 _GNU_SOURCE ARM_MATH_LOOPUNROLL)
  if(ARM_REQUIRE_MVE)
    target_compile_definitions(tiled_attention PRIVATE TILED_ATTENTION_REQUIRE_MVE)
  endif()
  if(ARM_PROFILE_CYCLES)
    target_compile_definitions(tiled_attention PRIVATE TILED_ATTENTION_PROFILE_CYCLES)
  endif()
  target_compile_options(
    tiled_attention
    PRIVATE
    -mcpu=cortex-m55
    -mthumb
    -mfloat-abi=${ARM_FLOAT_ABI}
    -Ofast
    -flax-vector-conversions
    -ffunction-sections
    -fdata-sections
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_options(
    tiled_attention
    PRIVATE
    -mcpu=cortex-m55
    -mthumb
    -mfloat-abi=${ARM_FLOAT_ABI}
    --specs=nano.specs
    --specs=nosys.specs
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/tiled_attention.map"
    "-T${CMAKE_CURRENT_SOURCE_DIR}/platform/qemu-mps3-an547/qemu_mps3_an547.ld"
  )
  target_link_libraries(tiled_attention PRIVATE m c gcc nosys)

  find_program(QEMU_SYSTEM_ARM qemu-system-arm)
  if(QEMU_SYSTEM_ARM)
    add_custom_target(
      run_qemu
      COMMAND "${QEMU_SYSTEM_ARM}"
              -machine mps3-an547
              -cpu cortex-m55
              -nographic
              -serial null
              -monitor null
              -semihosting-config enable=on,target=native
              -kernel "$<TARGET_FILE:tiled_attention>"
      DEPENDS tiled_attention
      USES_TERMINAL
      VERBATIM
    )
  endif()
else()
  target_compile_options(tiled_attention PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_link_libraries(tiled_attention PRIVATE CMSISDSP m)
endif()

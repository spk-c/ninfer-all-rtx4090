# Static context-cost calibrator. Transfer measurements call production Paged-KV copy primitives,
# then release their synthetic fixtures before the public Engine optionally loads one real artifact
# for text/Vision prefill measurements.
ninfer_add_bench(ninfer_context_cost_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_cost_bench.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/context_cost_measure.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/model_context_fixture.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/transfer_context_fixture.cpp"
  LIBRARIES ninfer_engine ninfer_core ${NINFER_CUDART_TARGET} ninfer::json)
target_include_directories(ninfer_context_cost_bench PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ninfer_context_cost_bench PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Machine transfer calibration must not depend on building every model/kernel.
# It shares the exact production KV transfer fixture and report/fit implementation.
add_executable(ninfer_context_transfer_bench
  context_cost/context_cost_bench.cpp
  context_cost/context_cost_measure.cpp
  context_cost/transfer_context_fixture.cpp)
ninfer_internal_includes(ninfer_context_transfer_bench)
target_compile_definitions(ninfer_context_transfer_bench PRIVATE NINFER_CONTEXT_TRANSFER_ONLY=1)
target_link_libraries(ninfer_context_transfer_bench PRIVATE ninfer_core ninfer_context_cost ninfer::json)

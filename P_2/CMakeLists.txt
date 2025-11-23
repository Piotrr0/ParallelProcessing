cmake_minimum_required(VERSION 3.15)
project(P VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

get_filename_component(hw_proto "protos/matrix.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

set(PROTO_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(hw_proto_srcs "${PROTO_OUT_DIR}/matrix.pb.cc")
set(hw_proto_hdrs "${PROTO_OUT_DIR}/matrix.pb.h")
set(hw_grpc_srcs "${PROTO_OUT_DIR}/matrix.grpc.pb.cc")
set(hw_grpc_hdrs "${PROTO_OUT_DIR}/matrix.grpc.pb.h")

add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${PROTO_OUT_DIR}"
           --cpp_out "${PROTO_OUT_DIR}"
           -I "${hw_proto_path}"
           --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
           "${hw_proto}"
      DEPENDS "${hw_proto}")

include_directories("${PROTO_OUT_DIR}")

add_executable(server src/server.cpp ${hw_proto_srcs} ${hw_grpc_srcs})
target_link_libraries(server ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})

add_executable(client src/client.cpp src/client_cli.cpp ${hw_proto_srcs} ${hw_grpc_srcs})
target_link_libraries(client ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
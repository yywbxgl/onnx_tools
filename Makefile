
CFLAGS:=  -std=c++11 -pthread  -I/usr/local/include  -I./  -g


.PHONY: all traget  onnx_parse  onnx_parse_weight shape_inference optimizer  operator_convert clean  inception_convert

all:
	$(MAKE) -j8 traget

traget: onnx_parse onnx_parse_weight  shape_inference optimizer operator_convert

clean:
	rm bin/*

onnx_parse: 
	g++  onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/onnx_parse

onnx_parse_weight: 
	g++  onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -D RAW_DATA -o bin/onnx_parse_weight

shape_inference: 
	g++  shape_inference.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o  bin/shape_inference

optimizer: 
	g++  optimizer.cpp  /usr/local/lib/libonnx.a   /usr/local/lib/libonnx_proto.a  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/optimizer

operator_convert: 
	g++  operator_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/operator_convert

inception_convert: 
	g++  inception_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/inception_convert
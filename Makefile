

CFLAGS:=  -std=c++11 -pthread  -I/usr/local/include  -I./ 


.PHONY: all  onnxParser  onnxParser_weight shapeInference optimizer  operatorConvert
all: onnxParser onnxParser_weight  shape_inference optimizer op_convert

onnxParser: 
	g++  onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/onnxParse

onnxParser_weight: 
	g++  onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -D RAW_DATA -o bin/onnxParse

shape_inference: 
	g++  shape_inference.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o  bin/shape_inference

optimizer: 
	g++  optimizer.cpp  /usr/local/lib/libonnx.a   /usr/local/lib/libonnx_proto.a  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/onnx_optimizer

op_convert: 
	g++  operator_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a  ${CFLAGS} -o bin/op_convert

clean:
	rm bin/*
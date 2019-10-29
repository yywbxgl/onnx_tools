# onnx tools
onnx parser && onnx operator convertor && onnx shape inference
and some python script

## Limit
[support operator limit](./doc/Operator-ykx-limit.md)

## environment
pip3 install protobuf==3.8.0 
pip3 install onnx==1.6.0


## Usage
```
# parse onnx model
./bin/onnxParse  models/model.onnx

# onnx operator convert
./bin/op_convert  input.onnx  output_name
```


## Complie command
```
# compile proto file 
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto

# onnx parser
g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o bin/onnxParse
g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -D RAW_DATA -o bin/onnxParse_weight

# onnx shape inference
g++ shape_inference.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a   -std=c++11 -pthread -I/usr/local/include -I./  -o  bin/shape_inference
g++ shape_inference2.cpp  onnx.pb.cc  /usr/local/lib/libonnx.a   /usr/local/lib/libonnx_proto.a  /usr/local/lib/libprotobuf.a   -std=c++11 -pthread -I/usr/local/include -I./  -o  bin/shape_inference2

# onnx optimizer
g++ optimizer.cpp   /usr/local/lib/libonnx.a   /usr/local/lib/libonnx_proto.a   /usr/local/lib/libprotobuf.a  -std=c++11 -pthread -I/usr/local/include -I./  -o  bin/onnx_optimizer

# onnx operator convertor
g++ operator_convert.cpp   onnx.pb.cc  /usr/local/lib/libprotobuf.a  -std=c++11 -pthread -I/usr/local/include  -I./  -o  bin/op_convert

```
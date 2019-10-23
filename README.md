# onnx tools
onnx parser && onnx operator convertor && onnx shape inference
and some python script

## Limit
[support operator limit](./doc/Operator-ykx-limit.md)

## environment
protobuf    v3.8.0  
onnx.proto  v1.6.0


## Usage
```
# parse onnx model
./bin/onnxParse  models/model.onnx

# onnx operator convert
./bin/op_convert  input.onnx  output_name

# check onnx model
python3 ./python_tool/cherck_model.py  out.onnx 

# run onnx model
python3 ./python_tool/onnx_run2.py  out.onnx

```


## Complie command
```
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto

g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o onnxParse

g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -D RAW_DATA -o onnxParse_weight

g++ op_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o op_convert

g++ main.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -I./ -o bin/op_convert
```
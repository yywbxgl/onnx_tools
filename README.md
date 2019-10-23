# onnx operator convertor


### environment
protobuf    v3.8.0
onnx.proto  v1.6.0


### Complie
```

protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto

g++ main.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -I./ -o bin/op_convert

```

### Usage

```
# onnx operator convert
./bin/op_convert  input.onnx  output_name

# check onnx model
python3 ./python_tool/cherck_model.py  out.onnx 

```

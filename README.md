# onnx tools
onnx parser && onnx operator convertor && onnx shape inference  
and some python script


## Limit
[support operator limit](./doc/Operator-ykx-limit.md)

## environment
```
pip3 install protobuf==3.8.0  
pip3 install onnx==1.5.0
```

## Usage
```shell
# compile
make

# parse onnx model
./bin/onnxParse  models/model.onnx

# onnx operator convert
./bin/op_convert  input.onnx  output_name
```



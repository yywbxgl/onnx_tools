#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 
 

#include "onnx.pb.h"
// #include "include/json/json.h"

using namespace std;

// #define RAW_DATA 
#define DOC_STRING 0

void get_weight (const onnx::GraphProto& graph_proto) {
  for(int i = 0; i < graph_proto.initializer_size(); i++) {
		const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
	  // const onnx::TensorProto_DataType& datatype = tensor_proto.data_type();
		int tensor_size  = 1;
  }
}

void get_layer_params (const onnx::NodeProto& node_proto) {
  // for (int i = 0; i < node_proto.input_size(); i++) {
  //   std::cout << "Input: " << node_proto.input(i) << '\n';
  // }
  // for (int i = 0; i < node_proto.output_size(); i++) {
  //   std::cout << "Output: " << node_proto.output(i) << '\n';
  // }

  std::string layer_type = node_proto.op_type();
  if(layer_type == "Conv") {
    std::cout << "_______convolution layer_______" << '\n';
    int pad_h, pad_w;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int dilation_h, dilation_w;

    int group = -1;

  std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
  for(int i = 0; i < node_proto.attribute_size(); i++) {
    const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
    std::string attribute_name = attribute_proto.name();
    std::cout << "attribute[" << i << "] = " << attribute_name << '\n';

// TO DO: add groups and bias to hyperparams

      if(attribute_name == "dilations") {
        dilation_h = attribute_proto.ints(0);
        std::cout << "dilation height: " << dilation_h << '\n';
        dilation_w = attribute_proto.ints(1);
        std::cout << "dilation widht: " << dilation_w << '\n';
      } else if(attribute_name == "group") {
       std::cout << attribute_proto.ByteSizeLong() << '\n';
       group = attribute_proto.i();
       std::cout << "group: " << group << '\n';
      } else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        std::cout << "kernel height: " << kernel_h << '\n';
        kernel_w = attribute_proto.ints(1);
        std::cout << "kernel width: " << kernel_w << '\n';
      } else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
        std::cout << "pad height: " << pad_h << '\n';
        pad_w = attribute_proto.ints(1);
        std::cout << "pad width: " << pad_w << '\n';
      } else if(attribute_name == "strides") {
        stride_h = attribute_proto.ints(0);
        std::cout << "stride height: " << stride_h << '\n';
        stride_w = attribute_proto.ints(1);
        std::cout << "stride width: " << stride_w << '\n';
      }

    }
  }
  else if(layer_type == "MaxPool") {
    std::cout << "_______maxpooling layer_______" << '\n';
    int pad_h, pad_w;
    int stride_h, stride_w;
    int kernel_h, kernel_w;

    std::cout << "quantity of attribute = " << node_proto.attribute_size() << '\n';
    for(int i = 0; i < node_proto.attribute_size(); i++) {
      const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
      std::string attribute_name = attribute_proto.name();
      std::cout << "attribute[" << i << "] = " << attribute_name << '\n';

      if(attribute_name == "strides") {
        stride_h = attribute_proto.ints(0);
        std::cout << "stride height: " << stride_h << '\n';
        stride_w = attribute_proto.ints(1);
        std::cout << "stride width: " << stride_w << '\n';
      }
      else if(attribute_name == "pads") {
        pad_h = attribute_proto.ints(0);
        std::cout << "pad height: " << pad_h << '\n';
        pad_w = attribute_proto.ints(1);
        std::cout << "pad width: " << pad_w << '\n';
      }
      else if(attribute_name == "kernel_shape") {
        kernel_h = attribute_proto.ints(0);
        std::cout << "kernel height: " << kernel_h << '\n';
        kernel_w = attribute_proto.ints(1);
        std::cout << "kernel width: " << kernel_w << '\n';
      }
    }
  }
}


void parse_tensor_proto(const onnx::TensorProto& tensor_proto, bool raw_data=false)
{
  cout << "name: " << tensor_proto.name() << endl;
  cout << "dims: ";
  for (int i =0; i< tensor_proto.dims_size(); ++i)
  {
    cout << tensor_proto.dims(i) << " ";
  }
  cout << endl;
  std::string date_type = onnx::TensorProto::DataType_Name(tensor_proto.data_type());
  cout << "date_type: " << date_type << endl;
  //cout << "segment begin: " << tensor_proto.segment().begin() << endl;
  //cout << "segment end: " << tensor_proto.segment().end() << endl;

#ifdef RAW_DATA
  raw_data = true;
#endif

if (raw_data)
{
//todo  只解析了float32 与 int64两种数据类型
  if (date_type == "FLOAT")
  {
    cout << "float_data = [" << endl;
    for (int i =0; i< tensor_proto.float_data_size(); ++i)
    {
      cout << tensor_proto.float_data(i) << " ";
    }
    cout << "]" << endl;
  }
  else if (date_type == "DOUBLE")
  {
    cout << "int64_data = [" << endl;
    for (int i =0; i< tensor_proto.double_data_size(); ++i)
    {
      cout << tensor_proto.double_data(i) << " ";
    }
    cout << "]" << endl;
  }
  else if (date_type == "INT64")
  {
    cout << "int64_data = [" << endl;
    for (int i =0; i< tensor_proto.int64_data_size(); ++i)
    {
      cout << tensor_proto.int64_data(i) << " ";
    }
    cout << "]" << endl;
  }
  else if (date_type == "UINT64")
  {
    cout << "int64_data = [" << endl;
    for (int i =0; i< tensor_proto.uint64_data_size(); ++i)
    {
      cout << tensor_proto.uint64_data(i) << " ";
    }
    cout << "]" << endl;
  }
  else if (date_type == "INT32")
  {
    cout << "int64_data = [" << endl;
    for (int i =0; i< tensor_proto.int32_data_size(); ++i)
    {
      cout << tensor_proto.int32_data(i) << " ";
    }
    cout << "]" << endl;
  }
  else
  {
    cout << "warning: can not pares data!" << endl;
  }

  if (tensor_proto.raw_data().length() > 0)
  {
    cout << "raw_data=[ " << endl;
    
    for (int i=0; i < tensor_proto.raw_data().length(); ++i)
    {
      // 显示二位  自动填充0
      cout <<  std::hex << std::setw(2) << std::setfill('0') << (uint16_t)tensor_proto.raw_data()[i] << " ";

    }

    cout << "]" << endl;
  }
}

  cout << endl;
}


void parse_valueinfo_proto(const onnx::ValueInfoProto& valueinfo_proto)
{
  cout << "name: " << valueinfo_proto.name() << endl;
  if (DOC_STRING){
    cout << "doc_string: " << valueinfo_proto.doc_string() << endl;
    cout << "type_denotaion: " << valueinfo_proto.type().denotation() << endl;
  }
  cout << "type_value_elem_type: " << onnx::TensorProto::DataType_Name(valueinfo_proto.type().tensor_type().elem_type()) << endl;
  cout << "type_value_shape_dim_value = [ " ;
  for (int i=0; i < valueinfo_proto.type().tensor_type().shape().dim_size(); ++i)
  {
    if (DOC_STRING){
      cout << "type_value_shape_dim_dim_value: " << valueinfo_proto.type().tensor_type().shape().dim(i).dim_value() << endl;
      cout << "type_value_shape_dim_dim_param: " << valueinfo_proto.type().tensor_type().shape().dim(i).dim_param() << endl;
      cout << "type_value_shape_dim_denotation: " << valueinfo_proto.type().tensor_type().shape().dim(i).denotation() << endl;
    }else{
      cout <<  valueinfo_proto.type().tensor_type().shape().dim(i).dim_value();
      if (valueinfo_proto.type().tensor_type().shape().dim(i).dim_value() == 0)
      {
        cout << "/" << valueinfo_proto.type().tensor_type().shape().dim(i).dim_param();
      }

      cout << "  ";
    }
  }
  cout << "]" << endl;

  cout << endl;
   
}

// void parse_TensorAnnotation(const onnx::TensorAnnotation& tensorAnnotation_proto)
// {
//   cout << "tensor_name: " << tensorAnnotation_proto.tensor_name() << endl;
//   for (int i =0; i< tensorAnnotation_proto.quant_parameter_tensor_names_size(); ++i)
//   {
//     cout << "tensor_name_key: " << tensorAnnotation_proto.quant_parameter_tensor_names(i).key() << endl;
//     cout << "tensor_name_value: " << tensorAnnotation_proto.quant_parameter_tensor_names(i).value() << endl;
//   }
//   cout <<endl;

// }

void parse_attribute_proto(const onnx::AttributeProto& attr_proto)
{
  cout << "name: " << attr_proto.name() << endl;
  if (DOC_STRING){
    cout << "ref_attr_name: " << attr_proto.ref_attr_name() << endl;
    cout << "doc_string: " << attr_proto.doc_string() << endl;
  }
  std::string attr_type = onnx::AttributeProto::AttributeType_Name(attr_proto.type());
  cout << "type: " <<  attr_type << endl;
  
  // todo 只解析了部分类型
  if (attr_type == "INT")
  {
    cout << "Value = " << attr_proto.i() <<endl;
  }
  else if  (attr_type == "INTS")
  {
    // cout << "Value_size= " << attr_proto.ints_size() << endl;
    cout << "Value = [" ;
    for (int i=0; i < attr_proto.ints_size(); ++i)
    {
      cout << attr_proto.ints(i) << " ";
    }
    cout << "]" << endl;
  }   
  else if (attr_type == "STRING")
  {
    cout << "Value = " << attr_proto.s() <<endl;
  }
  else if  (attr_type == "STRINGS")
  {
    cout << "Value = [" ;
    for (int i=0; i < attr_proto.strings_size(); ++i)
    {
      cout << attr_proto.strings(i) << " ";
    }
    cout << "]" << endl;
  }
  else if (attr_type == "FLOAT")
  {
    cout << "Value = " << attr_proto.f() <<endl;
  }
  else if  (attr_type == "FLOATS")
  {
    cout << "Value = [" ;
    for (int i=0; i < attr_proto.floats_size(); ++i)
    {
      cout << attr_proto.floats(i) << " ";
    }
    cout << "]" << endl;
  }
  else if  (attr_type == "TENSOR")
  {
    parse_tensor_proto(attr_proto.t(), true);
  }
  else 
  {
      cout << "warning: can not parse data!" << endl;
  }
  cout << endl;
  
}

void parse_node(const onnx::NodeProto& node_proto)
{
  // cout << "----------model graph node [  "  << node_proto.name() << " ]-------------" << endl;
  cout << "name: " << node_proto.name()  << endl;
  cout << "op_type: " << node_proto.op_type() << endl;
  if (DOC_STRING){
    cout << "domain: " << node_proto.domain() << endl;
    cout << "doc_string: " << node_proto.doc_string() << endl;
  }

  cout << "input = ";
  for (int i =0; i< node_proto.input_size(); ++i)
  {
    cout << node_proto.input(i) << "   ";
  }
  cout << endl;

  cout << "out = ";
  for (int i =0; i< node_proto.output_size(); ++i)
  {
    cout << node_proto.output(i) << "   ";
  }
  cout << endl;

  cout << "attribute = [" <<  endl;
  for (int i=0 ; i< node_proto.attribute_size(); ++i)
  {
    parse_attribute_proto(node_proto.attribute(i));
  }
  cout << "]" << endl;

  cout<< endl <<endl;

}

void parse_graph_proto(const onnx::GraphProto& graph_proto)
{
  onnx::NodeProto node_proto;
  cout << "----------model graph------------- "  << endl;
  cout << "name: " << graph_proto.name() << endl;
#ifdef DOC_STRING
  cout << "doc_string: " << graph_proto.doc_string() << endl;
#endif

  // 解析 initializer 
  cout << "----------model graph initializer ------------- "  << endl;
  cout << "initializer = [" << endl;
  for (int i =0; i< graph_proto.initializer_size(); ++i)
  {
    parse_tensor_proto(graph_proto.initializer(i));
  }
  cout << "]" << endl;

  // 解析 input, 包括各个层的input参数 
  cout << "----------model graph input ------------- "  << endl;
  cout << "input = [" << endl;
  for (int i =0; i< graph_proto.input_size(); ++i)
  {
    parse_valueinfo_proto(graph_proto.input(i));
  }
  cout << "]" << endl;


  // 解析 output，只有输出层
  cout << "----------model graph output ------------- "  << endl;
  cout << "output = [" << endl;
  for (int i =0; i< graph_proto.output_size(); ++i)
  {
    parse_valueinfo_proto(graph_proto.output(i));
  }
  cout << "]" << endl;


   // 解析 value_info， 中间层的output
  cout << "----------model graph value_info ------------- "  << endl;
  cout << "value_info = [" << endl;
  for (int i =0; i< graph_proto.value_info_size(); ++i)
  {
    parse_valueinfo_proto(graph_proto.value_info(i));
  }
  cout << "]" << endl;

  // 解析 quantization_annotation ？
  cout << "----------model graph quantization_annotation ------------- "  << endl;
  cout << "quantization_annotation = [" << endl;
  // for (int i =0; i< graph_proto.quantization_annotation_size(); ++i)
  // {
  //   parse_TensorAnnotation(graph_proto.quantization_annotation(i));
  // }
  cout << "]" << endl;

  // 解析 Nodes  中间层节点
  cout << "----------model graph Nodes ------------- "  << endl;
  cout << "Nodes = [" << endl << endl;
  for (int i =0; i< graph_proto.node_size(); ++i)
  {
    cout << "____node " << i << " ____" << endl;
    parse_node(graph_proto.node(i));
  }
  cout << "]" << endl;
}


// 解析 Model_proto
void parse_onnx_model(const onnx::ModelProto& model_proto) 
{
  // Json::Value model_info;
  // model_info["ir_version"] = Json::Value(int(model_proto.ir_version()));

  cout << "----------model------------- "  << endl;
  cout << "ir_version: " << model_proto.ir_version() << endl;
  cout << "opset_import = [" << endl;
  for (int i=0; i<model_proto.opset_import_size(); ++i){
    cout << "    domain: " << model_proto.opset_import(i).domain() << endl;
    cout << "    version: " << model_proto.opset_import(i).version()  << endl;
  }
  cout << "]" << endl;
  cout << "producer_name: " << model_proto.producer_name() << endl;
  cout << "producer_version: " << model_proto.producer_version() << endl;
  cout << "domain: " << model_proto.domain() << endl;
  cout << "model_version: " << model_proto.model_version() << endl;
  cout << "doc_string: " << model_proto.doc_string() << endl; 
  cout << "metadata_props = [" << endl;
  for (int i=0; i<model_proto.metadata_props_size(); ++i){
    cout << "    key: " << model_proto.metadata_props(i).key() << endl;
    cout << "    value: " << model_proto.metadata_props(i).value()  << endl;
  }
  cout << "]" << endl;

  if(model_proto.has_graph()) {
    parse_graph_proto(model_proto.graph());
  }

}



int main(int argc, char const *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " ONNX_FILE" << '\n';
    return -1;
  }

  onnx::ModelProto model_proto;


  {
    std::fstream input(argv[1], ios::in | ios::binary);
    if (!input) {
      std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
    } else if (!model_proto.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx model." << std::endl;
      return -1;
    }
  }

  parse_onnx_model(model_proto);

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

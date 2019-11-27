#include "onnx.pb.h"
#include "optimizer/eliminate_one_node.h"
#include "optimizer/fuse_matmul_add_bias_into_gemm.h"
#include "optimizer/convert_autoPad_to_pads.h"
#include "optimizer/convert_flatten_to_reshape.h"
#include "optimizer/convert_shape_to_inilizter.h"
#include "optimizer/convert_constant_to_initializer.h"

using namespace std;


// 添加一个softmax节点在网络末尾
void add_softmax_node(onnx::ModelProto& model_proto)
{
    //找到最后一个节点
    cout << "---- add softmax at the end of network" << endl;
    std::string output_name = model_proto.graph().output(0).name();

    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if(it->output(0) == output_name)
        {
            std::string pre_node = it->name();
            cout << "find end node: " << pre_node << " " << it->op_type() << endl;
            it->set_output(0, pre_node + "out");
            
            // 新增softmax node
            std::string new_node_name = "softmax_" + output_name;
            onnx::NodeProto* new_node = model_proto.mutable_graph()->add_node();
            new_node->set_name(new_node_name);
            new_node->set_op_type("Softmax");
            new_node->add_input(pre_node + "out");
            new_node->add_output(output_name);
            break;
        }
    }
    return;
}


// 修改input形状
void modify_input(onnx::ModelProto& model_proto)
{
    //修改producer_name
    std::string* name = model_proto.mutable_producer_name();
    *name = "sun-smile";
    cout << "---- modify producer_name, name = " << model_proto.producer_name() << endl;

    //修改input
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(1).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(2).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(3).dim_value() << endl;

    cout << "---- modify input shape " << endl;
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(3);
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(224);
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(3)->set_dim_value(224);

    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(1).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(2).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(3).dim_value() << endl;

    //修改output
    // cout << "modele output size " << model_proto.graph().output(0).type().tensor_type().shape().dim_size() << endl;
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    cout << "modele output " << model_proto.graph().output(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele output " << model_proto.graph().output(0).type().tensor_type().shape().dim(1).dim_value() << endl;

    return;
}



// 修改模型数据内容
void modify_weights(onnx::ModelProto& model_proto)
{
    for (int i =0; i < model_proto.graph().initializer_size(); ++i)
    {
        if (model_proto.graph().initializer(i).data_type() ==  1)
        {
            model_proto.mutable_graph()->mutable_initializer(i)->set_float_data(0, 0);
            cout << model_proto.graph().initializer(i).name() << endl;
            cout <<  model_proto.graph().initializer(i).float_data(0) << endl;
        }
    }

    return;
}


// 修改softmax layer 的属性，axis属性修改为默认值1
void eliminate_softmax_attributes(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->op_type() == "Softmax")
        {
            cout << "---sun--- elimate Softmax axis" << endl;
            if (it->attribute_size() != 0)
            {
                //只有一个aixs属性，消除后变为默认值1
                it->clear_attribute();
            }
        }
    }
}


// 修改IR_version
void modify_opset(onnx::ModelProto& model_proto)
{
    // model_proto.mutable_opset_import(0)->set_version(8);
    model_proto.set_ir_version(6);

    cout << "----------model------------- "  << endl;
    cout << "ir_version: " << model_proto.ir_version() << endl;
    cout << "opset_import = [" << endl;
    for (int i=0; i<model_proto.opset_import_size(); ++i){
        cout << "    domain: " << model_proto.opset_import(i).domain() << endl;
        cout << "    version: " << model_proto.opset_import(i).version()  << endl;
    }
    return;
}

// 删除所有value_info保存的信息
void clear_value_info(onnx::ModelProto& model_proto)
{
    model_proto.mutable_graph()->clear_value_info();
}


int main(int argc, char const *argv[]) 
{

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " ONNX_FILE  " <<  " OUTPUT_FILE"  << '\n';
        return -1;
    }

    onnx::ModelProto model_proto;

    {
        // Read the existing model.
        std::fstream input_model(argv[1], ios::in | ios::binary);
        if (!input_model) 
        {
            std::cout << argv[1] << ": file not found." << '\n';
        } 
        else if (!model_proto.ParseFromIstream(&input_model)) 
        {
            std::cerr << "Failed to parse onnx model." << std::endl;
            return -1;
        }
    }

    // onnx convert
    // eliminate_node(model_proto, "Identity");
    // eliminate_node(model_proto, "Dropout");
    // eliminate_node(model_proto, "ReduceMean");
    // eliminate_node(model_proto, "Transpose");
    // fuse_matmul_add_bias_into_gemm(model_proto);
    // auto_pad_convert(model_proto);
    // modify_input(model_proto);
    // eliminate_softmax_attributes(model_proto);
    // modify_weights(model_proto);
    // clear_value_info(model_proto);
    // add_softmax_node(model_proto);


    // convert_flatten_to_reshape(model_proto);
    // convert_shape_to_inilizter(model_proto);
    // eliminate_node(model_proto, "Shape");
    convert_constant_to_initializer(model_proto);

    // modify_opset(model_proto);

    {
        // Write the new model back to disk.
        std::string output_file_name  = std::string(argv[2]) + ".onnx";
        std::fstream output_model(output_file_name, ios::out | ios::trunc | ios::binary);
        if (!model_proto.SerializeToOstream(&output_model)) 
        {
            std::cerr << "Failed to write address book." << endl;
            return -1;
        }
    }


    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

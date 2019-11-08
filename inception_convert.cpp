#include "onnx.pb.h"
using namespace std;


void convert_maxpool_to_avagepool(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->op_type() == "MaxPool")
        {
            cout << "---sun--- convert MaxPool to AveragePool " << endl;
            it->set_op_type("AveragePool");
        }
    }
}


// 修改opset
void modify_opset(onnx::ModelProto& model_proto)
{
    model_proto.mutable_opset_import(0)->set_version(8);

    cout << "----------model------------- "  << endl;
    cout << "ir_version: " << model_proto.ir_version() << endl;
    cout << "opset_import = [" << endl;
    for (int i=0; i<model_proto.opset_import_size(); ++i){
        cout << "    domain: " << model_proto.opset_import(i).domain() << endl;
        cout << "    version: " << model_proto.opset_import(i).version()  << endl;
    }
    return;
}



bool find_input_element(onnx::ModelProto& model_proto, std::string name)
{
    //printf("to find_input_element %s\n", name.c_str());
    for (int i=0; i< model_proto.graph().input_size(); ++i)
    {
        //printf("it->name=%s\n", it->name().c_str());
        if (model_proto.graph().input(i).name() == name)
        {
            cout << "find input elemet " << model_proto.graph().input(i).name() << endl;
            return true;
        }
    }
 
    cout << "can not find input elemet " << name << endl;
    return false;
}

void remove_input_element(onnx::ModelProto& model_proto, std::string name)
{
    google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_input()->begin();
    for (; it != model_proto.mutable_graph()->mutable_input()->end(); ++it)
    {
        if (it->name() == name)
        {
            //删除当前node
            model_proto.mutable_graph()->mutable_input()->erase(it);
            printf("eliminate current input [%s] success.\n", name.c_str()); 
            break;
        }
    }

    return;
}


bool find_input_existed(onnx::ModelProto& model_proto, std::string name)
{
    //printf("to find input %s\n ", name.c_str());
    if (name == "data_0"){
        return true;
    }
    for (int i=0; i< model_proto.graph().node_size(); ++i)
    {
        if (model_proto.graph().node(i).output(0) == name)
        {
            //cout << "find "  << endl;
            return true;
        }
    }
 
    cout << "not find input " << name  << endl;
    return false;
}

void eliminate_deadend_node(onnx::ModelProto& model_proto)
{
    bool finish_flag = false;
    while(!finish_flag)
    {
        google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
        it = model_proto.mutable_graph()->mutable_node()->begin();

        for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
        {
            std::string input = it->input(0);
            if (find_input_existed(model_proto, input) == false)
            {
                //删除当前node
                printf("eliminate current node. input=%s  output=%s\n", 
                    it->input(0).c_str(), it->output(0).c_str()); 
                model_proto.mutable_graph()->mutable_node()->erase(it);

                //删除input
                // if (it->output_size() != 0)
                // {
                //     //printf("remove input element %s  %d\n", it->output(0).c_str(), it->output_size());
                //     remove_input_element(model_proto, it->output(0));
                // }

                break;
            }
        }

        if (it == model_proto.mutable_graph()->mutable_node()->end())
        {
            finish_flag = true;
        }
    }
}


void modify_input_output(onnx::ModelProto& model_proto)
{
    //修改output
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(192);
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(28);
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(3)->set_dim_value(28);
    // model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(28);
    // model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(28);
    cout << "output shape: "<< model_proto.graph().output(0).type().tensor_type().shape().dim(0).dim_value() << " " << 
        model_proto.graph().output(0).type().tensor_type().shape().dim(1).dim_value() << " " <<
        model_proto.graph().output(0).type().tensor_type().shape().dim(2).dim_value() << " " << 
        model_proto.graph().output(0).type().tensor_type().shape().dim(3).dim_value() << endl;
}


void  eliminate_inception_nodes(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->input(0) == "conv2/norm2_1")
        {
            cout << "find maxpooling node" << endl;
            // 修改output

            it->clear_output();
            it->add_output("prob_1");

            remove_input_element(model_proto, it->output(0));
        }

        if (it->input(0) == "loss3/classifier_1")
        {
            cout << "find maxpooling node" << endl;
            // 修改output

            it->clear_output();
            it->add_output("softmax_output");
        }
    }


    modify_input_output(model_proto);

}



void change_pool_pad(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        // if (it->op_type() == "MaxPool")
        if (it->op_type() == "MaxPool")
        {
            for (int i =0; i< it->attribute_size(); ++i)
            {
                if (it->attribute(i).name()== "pads")
                {
                    cout << "convert pad to 0,0,3,3" << endl;
                    it->mutable_attribute(i)->clear_ints();
                    it->mutable_attribute(i)->add_ints(0);
                    it->mutable_attribute(i)->add_ints(0);
                    it->mutable_attribute(i)->add_ints(2);
                    it->mutable_attribute(i)->add_ints(2);
                }
            }
        }
    }
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
            std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
        } 
        else if (!model_proto.ParseFromIstream(&input_model)) 
        {
            std::cerr << "Failed to parse onnx model." << std::endl;
            return -1;
        }
    }


    //eliminate_inception_nodes(model_proto);
    //dump_input_element(model_proto);
    //eliminate_deadend_node(model_proto);

    change_pool_pad(model_proto);
    modify_input_output(model_proto);
    clear_value_info(model_proto);

    {
        // Write the new model back to disk.
        std::string output_file_name  = std::string(argv[2]) + ".onnx";
        fstream output_model(output_file_name, ios::out | ios::trunc | ios::binary);
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

#include "onnx.pb.h"
using namespace std;

// 消除onnx model中 shape opearator 
bool convert_shape_to_inilizter(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->op_type() == "Shape")
        {
            // step1. 查找input的shape
            cout << "---- find Shape node:" << it->name() << endl;
            std::string shape_input = it->input(0);
            std::string shape_output = it->output(0);
            cout << "Shape input: " << shape_input << endl;
            std::vector<int> shape_input_shape;

            for (int i =0; i < model_proto.graph().value_info_size(); ++i)
            {
                if ( model_proto.graph().value_info(i).name() == shape_input)
                {
                    for (int j =0; j < model_proto.graph().value_info(i).type().tensor_type().shape().dim_size(); ++j)
                    {
                        shape_input_shape.push_back(model_proto.graph().value_info(i).type().tensor_type().shape().dim(j).dim_value());
                    }
                }
            }
            if (shape_input_shape.size() != 0)
            {
                cout << "shape input shape: " ;
                for (auto n :shape_input_shape)
		            cout << n << " ";
                cout  << endl;
            }
            else
            {
                cout << "!!! error. can not find the shape input shape" << endl;
                return false;
            }


            // step2. 新增shape参数对象，放入inilizater 与 input
            std::string shape_init =  shape_input + "shape_init";
            onnx::TensorProto* new_shape = model_proto.mutable_graph()->add_initializer();
            new_shape->set_name(shape_init);
            new_shape->set_data_type(onnx::TensorProto_DataType_INT64);  //data_type = INT64
            new_shape->add_dims(shape_input_shape.size());
            for (auto n :shape_input_shape)
            {
                new_shape->add_int64_data(n);
            }

            onnx::ValueInfoProto* new_input = model_proto.mutable_graph()->add_input();  
            new_input->set_name(shape_init);
            new_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(shape_input_shape.size());
            new_input->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);
            cout << "add shape inilizer success." << endl;

            // step3. 修改下一层node的input, 并删除当前node
            for (int i =0; i < model_proto.graph().node_size(); ++i)
            {
                if ( model_proto.graph().node(i).input_size()!=0  && model_proto.graph().node(i).input(0) == shape_output)
                {
                    cout << "find next node: " << model_proto.graph().node(i).name() << endl;
                    model_proto.mutable_graph()->mutable_node(i)->set_input(0, shape_init);
                }
            }

            model_proto.mutable_graph()->mutable_node()->erase(it);
            cout << "eliminate current node success."  << endl << endl;
            break;
        }
    }

    return true;
}
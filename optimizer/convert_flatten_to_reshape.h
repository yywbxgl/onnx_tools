#include "onnx.pb.h"
using namespace std;

// 将conv等op中的auto_pad属性修改为padding属性
bool convert_flatten_to_reshape(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->op_type() == "Flatten")
        {
            // step1. 查找input的shape
            cout << "---- find Flatten node:" << it->name() << endl;
            std::string flatten_input = it->input(0);
            cout << "flatten input: " << flatten_input << endl;
            std::vector<int> flatten_input_shape;

            for (int i =0; i < model_proto.graph().value_info_size(); ++i)
            {
                if ( model_proto.graph().value_info(i).name() == flatten_input)
                {
                    for (int j =0; j < model_proto.graph().value_info(i).type().tensor_type().shape().dim_size(); ++j)
                    {
                        flatten_input_shape.push_back(model_proto.graph().value_info(i).type().tensor_type().shape().dim(j).dim_value());
                    }
                }
            }
            if (flatten_input_shape.size() != 0)
            {
                cout << "flatten input shape: " ;
                for (auto n :flatten_input_shape)
		            cout << n << " ";
                cout  << endl;
            }
            else
            {
                cout << "!!! error. can not find the flatten input shape" << endl;
                return false;
            }

            //step2. 计算flatten转为reshape的参数
            int axis = 1;  // flatten默认值为1
            if (it->attribute_size() !=0 && it->attribute(0).name() == "axis")
            {
                axis = it->attribute(0).i();
            }

            printf("axis = %d\n", axis);

            if (axis >= flatten_input_shape.size())
            {
                printf("!!! error. axis is too large.\n");
                return false;
            }

            int a = 1;
            int b = 1;
            for (int i=0; i<axis; ++i)
            {
                a = a* flatten_input_shape[i];
            }
            for (int i=axis; i<flatten_input_shape.size(); ++i)
            {
                b = b*flatten_input_shape[i];
            }

            printf("resize shape: %d %d\n", a,b);
            
            // step3. 新增reshape参数对象，放入inilizater 与 input
            std::string reshape_weight =  flatten_input + "reshape_weight";
            onnx::TensorProto* new_shape = model_proto.mutable_graph()->add_initializer();
            new_shape->set_name(reshape_weight);
            new_shape->set_data_type(7);  //data_type = INT64
            new_shape->add_dims(2);
            new_shape->add_int64_data(a);  // todo 根据gemm B 参数  以及该层的output形状  来确认reshape形状
            new_shape->add_int64_data(b); 

            onnx::ValueInfoProto* new_input = model_proto.mutable_graph()->add_input();  
            new_input->set_name(reshape_weight);
            new_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
            new_input->mutable_type()->mutable_tensor_type()->set_elem_type(7);

            // step4. 修改flatten为resize
            it->set_op_type("Reshape");
            it->clear_attribute();
            it->add_input(reshape_weight) ;

            printf("convert flatten to resize\n");

        }
    }

    return true;
}
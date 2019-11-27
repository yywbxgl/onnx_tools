#include "onnx.pb.h"
using namespace std;

// 提取constant内容放到initializer中
bool convert_constant_to_initializer(onnx::ModelProto& model_proto)
{

    bool finish_flag = false;
    while(!finish_flag)
    {
        google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
        it = model_proto.mutable_graph()->mutable_node()->begin();

        for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
        {
            if (it->op_type() == "Constant")
            {
                // step1. find  Constant  node
                cout << "---- find constant node:" << it->name() << endl;
                // constant node 可能没有 name  以及input，
                std::string constant_output_name =  it->output(0);

                // step2. extracr  Constant  tensor to   initializer               
                for (int i=0; i< it->attribute_size(); ++i)
                {
                    //todo: 暂时值支持value , 不支持 sapre_value， value值的类型为tensor
                    if (it->attribute(i).name() == "value")
                    {
                        onnx::TensorProto* constant_tensor = model_proto.mutable_graph()->add_initializer();
                        constant_tensor->CopyFrom(it->attribute(i).t());
                        constant_tensor->set_name(constant_output_name);
                        cout << "====" << constant_tensor->dims_size() << endl;
                        for (auto i : constant_tensor->raw_data())
                        {
                            printf("%x ", i);
                        }
                        cout <<endl;

                        // 保持name不变 不需要添加input
                        // onnx::ValueInfoProto* new_input = model_proto.mutable_graph()->add_input();
                        // new_input->set_name(constant_output_name);
                        // cout << "data_type:" << it->attribute(i).t().data_type() << endl;
                        // new_input->mutable_type()->mutable_tensor_type()->set_elem_type(it->attribute(i).t().data_type());
                        // // cout << "has shape:" << new_input->mutable_type()->mutable_tensor_type()->has_shape() << endl;
                        // new_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
                        // // cout << "has shape:" << new_input->mutable_type()->mutable_tensor_type()->has_shape() << endl;
                        
                        // for (int j = 0; j < it->attribute(i).t().dims_size(); ++j)
                        // {
                        //     new_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(it->attribute(i).t().dims(j));
                        // }
                        // cout << "add new initializer and input success:" << constant_output_name << endl;
                        // cout << "has shape:" << new_input->mutable_type()->mutable_tensor_type()->has_shape() << endl;

                        // cout << "data type:" << constant_tensor->data_type() << endl;
                        // cout << "dims size:" << constant_tensor->dims_size() << endl;
                        // for (int j=0; j < constant_tensor->raw_data().length(); ++j)
                        // {
                        //     cout <<  std::hex << std::setw(2) << std::setfill('0') << (uint16_t)constant_tensor->raw_data()[j] << " ";
                        // }
                        
                    }
                }

               
                // step3. delete valueinfo and cureent node
                // google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>::iterator it_v_info; 
                // it_v_info = model_proto.mutable_graph()->mutable_value_info()->begin();

                // for (; it_v_info != model_proto.mutable_graph()->mutable_value_info()->end(); ++it_v_info)
                // {
                //     if (it_v_info->name() == constant_output_name)
                //     {
                //         //删除当前valueinfo
                //         model_proto.mutable_graph()->mutable_value_info()->erase(it_v_info);
                //         cout << "eliminate current value info success."  << endl;
                //         break;
                //     }
                // }

                //删除当前node
                model_proto.mutable_graph()->mutable_node()->erase(it);
                cout << "eliminate current node success."  << endl << endl;
                break;
            }
        }

        if(it == model_proto.mutable_graph()->mutable_node()->end())
        {
            finish_flag = true;
        }
    }

    return true;
}

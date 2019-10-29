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
                // step1. Constant
            }
        }
    }

    return true;
}
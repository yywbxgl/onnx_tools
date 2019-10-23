#include "onnx.pb.h"
using namespace std;

// 将conv等op中的auto_pad属性修改为padding属性
void auto_pad_convert(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        if (it->op_type() == "Conv" || it->op_type() == "AveragePool" || it->op_type() == "MaxPool")
        {
            for(int i=0; i < it->attribute_size(); ++i)
            {
                if (it->attribute(i).name() == "auto_pad")
                {
                    cout << "node name: " << it->name() << endl;
                    cout << "auto_pad:" << it->attribute(i).s() << endl;

                    if (it->mutable_attribute(i)->s() == "VALID")
                    {
                        it->mutable_attribute(i)->set_name("pads");
                        it->mutable_attribute(i)->set_type(onnx::AttributeProto_AttributeType_INTS);   //INTS
                        it->mutable_attribute(i)->clear_s();
                        it->mutable_attribute(i)->add_ints(0);
                        it->mutable_attribute(i)->add_ints(0);
                        it->mutable_attribute(i)->add_ints(0);
                        it->mutable_attribute(i)->add_ints(0);
                    }
                    else if (it->mutable_attribute(i)->s() == "SAME_UPPER" || it->mutable_attribute(i)->s() == "SAME_LOWER" )
                    {
                        it->mutable_attribute(i)->set_name("pads");
                        it->mutable_attribute(i)->set_type(onnx::AttributeProto_AttributeType_INTS);   //INTS
                        it->mutable_attribute(i)->clear_s();
                        int kernal_shape =-1;
                        int stride =-1;
                        for (int j=0; j < it->attribute_size(); ++j)
                        {
                            if (it->attribute(j).name() == "kernel_shape")
                            {
                                kernal_shape = it->attribute(j).ints(0); // todo  只支持正方形的kernal
                            }else if (it->attribute(j).name() == "strides")
                            {
                                stride = it->attribute(j).ints(0);  // todo  只支持正方形的stride
                            }
                        }
                        
                        cout << "kernal_shape=" << kernal_shape << ", stride=" << stride << endl;

                        if (kernal_shape <0 || stride <0)
                        {
                            cout << "error. can not find kernal_shape and stride." << endl;
                            return;
                        }

                        // todo  stride=1 或stride =2
                        int total_pad = -1;
                        int pad_left = -1;
                        int pad_right = -1;
                        // output_shape = ceil (input_size/stride)
                        total_pad = kernal_shape - stride;
                        pad_left = pad_right = total_pad/2;

                        if (total_pad % 2 == 1)
                        {
                            if (it->mutable_attribute(i)->s() == "SAME_UPPER")
                            {
                                // padding at the end for SAME_UPPER
                                // pad_right += 1;  
                                pad_left +=1;
                            }
                            else
                            {
                                // padding  at the beginning for SAME_LOWER
                                // pad_left += 1;
                                pad_right += 1; 
                            }
                        }

                        if (pad_left <0 || pad_left<0)
                        {
                            cout << "error. convert auto pad faild." << endl;
                            return;  
                        }

                        it->mutable_attribute(i)->add_ints(pad_left);
                        it->mutable_attribute(i)->add_ints(pad_left);
                        it->mutable_attribute(i)->add_ints(pad_right);
                        it->mutable_attribute(i)->add_ints(pad_right);

                        printf("convert auto_pad same to pads (%d,%d,%d,%d) \n", 
                            pad_left, pad_left, pad_right, pad_right);
                        
                    }
                }
            }
        }
    }

}
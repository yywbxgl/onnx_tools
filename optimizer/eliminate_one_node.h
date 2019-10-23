
#include "onnx.pb.h"
using namespace std;


//直接删除单层node  //todo:only support one input and one outpt
void eliminate_node(onnx::ModelProto& model_proto, std::string node_type)
{
    bool finish_flag = false;
    while(!finish_flag)
    {
        google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
        it = model_proto.mutable_graph()->mutable_node()->begin();

        for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
        {
            std::string input;
            std::string output;
            if (it->op_type() == node_type)
            {
                cout << "---- find node " << node_type << "----" << endl;
                cout << "name = " << it->name()  << endl;
                cout << "op_type = " << it->op_type() << endl;
                cout << "input = " << it->input(0) << endl;  // identify has only one input
                cout << "out = " << it->output(0) << endl;

                input = it->input(0);
                output = it->output(0);


                // 如果下一层是graph的output, 那么找到上一层，修改上层的ouput
                if (output == model_proto.graph().output(0).name())
                {
                    for (int i=0; i<model_proto.graph().node_size(); ++i) 
                    {
                        if (model_proto.graph().node(i).output(0) == input)
                        {
                            cout << "---- find prev node: " << endl;
                            cout << "name = " << model_proto.graph().node(i).name()  << endl;
                            cout << "op_type = " << model_proto.graph().node(i).op_type() << endl;
                            cout << "input = " << model_proto.graph().node(i).input(0) << endl; 
                            cout << "out = " << model_proto.graph().node(i).output(0) << endl;

                            cout << "modify the output to " << output << endl;
                            model_proto.mutable_graph()->mutable_node(i)->set_output(0, output);
                        }   
                    } 
                }
                else
                {
                    //找到下一层，修改下一层的input ps： only support one input and one outpt
                    for (int i=0; i<model_proto.graph().node_size(); ++i) 
                    {
                        if (model_proto.graph().node(i).input(0) == output)
                        {
                            cout << "---- find prev node: " << endl;
                            cout << "name = " << model_proto.graph().node(i).name()  << endl;
                            cout << "op_type = " << model_proto.graph().node(i).op_type() << endl;
                            cout << "input = " << model_proto.graph().node(i).input(0) << endl; 
                            cout << "out = " << model_proto.graph().node(i).output(0) << endl;

                            cout << "modify the input to " << input << endl;
                            model_proto.mutable_graph()->mutable_node(i)->set_input(0, input);
                        }   
                    }
                }

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
}
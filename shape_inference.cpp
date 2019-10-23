#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 
#include <vector>
#include <map>
#include <cmath>

#include "onnx.pb.h"

using namespace std;

class shapeInference
{
public:
    shapeInference(onnx::ModelProto &model_proto);

public:
    void init_shape_map();
    onnx::ModelProto inference();
    int node_inference(int node_index);
    int find_shape(std::string name);
    std::vector<int> get_shape(std::string name);
    int add_inference_shape(std::string name, std::vector<int> &shape, int type);
    bool get_broadcast_shape (std::vector<int> a, std::vector<int> b, std::vector<int> &result);
    int get_type(std::string name);
    int get_input_type();

private:

    std::vector<int> shape_inference_conv(int node_index);
    std::vector<int> shape_inference_broadcast(int node_index);
    std::vector<int> shape_inference_same(int node_index);
    std::vector<int> shape_inference_reshape(int node_index);
    std::vector<int> shape_inference_matmul(int node_index);
    std::vector<int> shape_inference_gemm(int node_index);
    std::vector<int> shape_inference_globalPooling(int node_index);
    std::vector<int> shape_inference_unsqueeze(int node_index);

public:
    onnx::ModelProto m_model_proto;
    std::map<std::string,std::vector<int>> m_shape_map;
    int m_input_type;
};

shapeInference::shapeInference(onnx::ModelProto &in_model_proto)
{
    m_model_proto = in_model_proto;
    init_shape_map();
    get_input_type();
}

void shapeInference::init_shape_map()
{
    for (int i=0; i< m_model_proto.graph().input_size(); ++i)
    {
        std::string shape_name = m_model_proto.graph().input(i).name();
        // cout << "find input " << shape_name << endl;
        std::vector<int> temp;
        for (int j=0; j < m_model_proto.graph().input(i).type().tensor_type().shape().dim_size(); ++j)
        {
            // cout <<  m_model_proto.graph().input(i).type().tensor_type().shape().dim(j).dim_value() << endl;
            temp.push_back(m_model_proto.graph().input(i).type().tensor_type().shape().dim(j).dim_value());
        }
        m_shape_map.insert(std::pair<std::string,std::vector<int>>(shape_name, temp));
    }

    for (int i=0; i<  m_model_proto.graph().value_info_size(); ++i)
    {
        std::string shape_name = m_model_proto.graph().value_info(i).name();
        // cout << "find value info " << shape_name << endl;
        std::vector<int> temp;
        for (int j=0; j < m_model_proto.graph().value_info(i).type().tensor_type().shape().dim_size(); ++j)
        {
            // cout <<  m_model_proto.graph().value_info(i).type().tensor_type().shape().dim(j).dim_value() << endl;
            temp.push_back(m_model_proto.graph().value_info(i).type().tensor_type().shape().dim(j).dim_value());
        }
        m_shape_map.insert(std::pair<std::string,std::vector<int>>(shape_name, temp));
    }

    return;
}

// ret -1未找到
int shapeInference::find_shape(std::string name)
{
    std::map<std::string,std::vector<int>>::iterator iter;
    iter = m_shape_map.find(name);
    if(iter != m_shape_map.end())
    {
        return 0;
    }
    else
    {
        return -1;
    }
}


std::vector<int> shapeInference::get_shape(std::string name)
{
    std::vector<int> result;
    std::map<std::string,std::vector<int>>::iterator iter;
    iter = m_shape_map.find(name);
    if(iter != m_shape_map.end())
    {
        result = iter->second;
        cout << "get shape success. " << name << endl;
    }

    for (int i=0; i<iter->second.size(); ++i)
    {
        cout << iter->second[i] << " " ;
    }
    cout << endl;

    return result;
}


int shapeInference::get_input_type()
{
    int ret=-1;
    for (int i=0; i< m_model_proto.graph().input_size(); ++i)
    {
        std::string name = m_model_proto.graph().input(i).name();
        if (name.find("input") != std::string::npos || name.find("Input") != std::string::npos || name.find("INPUT") != std::string::npos)
        {
            ret = m_model_proto.graph().input(i).type().tensor_type().elem_type();
        }
    }
    if (ret != -1)
    {
        m_input_type = ret;
        cout << "my input type: " << m_input_type << endl;
    }
    return ret;  
}

int shapeInference::get_type(std::string name)
{
    for (int i=0; i< m_model_proto.graph().input_size(); ++i)
    {
        if (name ==  m_model_proto.graph().input(i).name())
        {
            return m_model_proto.graph().input(i).type().tensor_type().elem_type();
        }
    }

    for (int i=0; i< m_model_proto.graph().initializer_size(); ++i)
    {
        if (name ==  m_model_proto.graph().initializer(i).name())
        {
            return m_model_proto.graph().initializer(i).data_type();
        }
    }

    return -1;
}

bool shapeInference::get_broadcast_shape(std::vector<int> a, std::vector<int> b, std::vector<int>  &result)
{
    int len_a = a.size() -1 ;
    int len_b = b.size() -1 ;
    std::vector<int> ret;
    while ( len_a >=0 || len_b >=0 )
    {
        int temp_a = 1;
        int temp_b = 1;
        if (len_a >=0) temp_a = a[len_a];
        if (len_b >=0) temp_b = b[len_b];
        
        if ((temp_a != temp_b) && !(temp_a ==1 || temp_b==1))
        {
            cout << "error!. can not bradcast. " <<  a[len_a] << " to " << b[len_b] << endl;
            return false;
        }
        ret.push_back(max(temp_a,  temp_b));
        len_a --;
        len_b --;
    }
    
    for (int i=ret.size()-1; i >=0; --i)
    {
        result.push_back(ret[i]);
    }

    return true;
}

int shapeInference::add_inference_shape(std::string name, std::vector<int> &shape, int type)
{
    onnx::ValueInfoProto* new_proto = m_model_proto.mutable_graph()->add_value_info();
    new_proto->set_name(name);
    new_proto->mutable_type()->mutable_tensor_type()->set_elem_type(type);
    cout << "add to shape map. " << name << endl;
    for (int i=0; i< shape.size(); ++i)
    {
        new_proto->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(shape[i]);
    }

    return 0;
}

// conv 推理
std::vector<int> shapeInference::shape_inference_conv(int node_index)
{

    //conv 输入为x w b(optional)
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());

    int N = -1;
    int C = -1;
    int H = -1;
    int W = -1;
    
    if (m_model_proto.graph().node(node_index).op_type() == "Conv")
    {
        std::string x_name = m_model_proto.graph().node(node_index).input(0);
        std::string w_name = m_model_proto.graph().node(node_index).input(1);
        
        // cout << x_name << endl;
        std::vector<int> x_shape = get_shape (x_name);
        std::vector<int> w_shape = get_shape (w_name);

        N = x_shape[0];
        C = w_shape[0];
        H = x_shape[2];
    }
    else 
    {
        std::string input_name = m_model_proto.graph().node(node_index).input(0);
        std::vector<int> input_shape = get_shape(input_name);
        N = input_shape[0];
        C = input_shape[1];
        H = input_shape[2];
    }

    int stride = -1;
    int left_pad = -1 ;
    int right_pad = -1 ;
    int kernal_shape = -1;
    bool auto_pad = false;
    std::string auto_pad_value;
    for (int i=0; i< m_model_proto.graph().node(node_index).attribute_size(); ++i)
    {
        if (m_model_proto.graph().node(node_index).attribute(i).name() == "strides")
        {
            stride = m_model_proto.graph().node(node_index).attribute(i).ints(0); //todo: only support same stride
            cout << "stride: " <<  stride << endl;
        }
        else if (m_model_proto.graph().node(node_index).attribute(i).name() == "kernel_shape")
        {
            kernal_shape = m_model_proto.graph().node(node_index).attribute(i).ints(0); //todo: only support same kernel_shape
            cout << "kernal_shape: " <<  kernal_shape << endl;
        }
        else if (m_model_proto.graph().node(node_index).attribute(i).name() == "pads")
        {
            left_pad = m_model_proto.graph().node(node_index).attribute(i).ints(0);  //todo: only support same pads
            right_pad = m_model_proto.graph().node(node_index).attribute(i).ints(2);
            cout << "pads: " <<  left_pad << " "<< right_pad << endl;
        }
        else if (m_model_proto.graph().node(node_index).attribute(i).name() == "auto_pad")
        {
            auto_pad = true;
            auto_pad_value = m_model_proto.graph().node(node_index).attribute(i).s();
            cout << "auto_pad: " <<  auto_pad_value  << endl;
        }
    }

    std::vector<int> output_shape;
    int out = -1;    

    if (auto_pad_value == "SAME_UPPER" || auto_pad_value == "SAME_LOWER")
    {
        out = ceil(H/(float)stride);
    }
    else if (auto_pad_value == "VALID") 
    {
        left_pad = right_pad = 0;
        out = floor((H + left_pad + right_pad - kernal_shape)/(float)stride) + 1;
    }
    else
    {
        out = floor((H + left_pad + right_pad - kernal_shape)/(float)stride) + 1;
    }
 

    cout << "out_shape: " << N << " " << C << " " << out << " " << out << endl;

    output_shape.push_back(N);
    output_shape.push_back(C);
    output_shape.push_back(out);
    output_shape.push_back(out);

    return output_shape;
}

// relu 等推理
std::vector<int> shapeInference::shape_inference_same(int node_index)
{
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());

    std::string input_name = m_model_proto.graph().node(node_index).input(0);
    std::string output_name = m_model_proto.graph().node(node_index).output(0);
    std::vector<int> input_shape = get_shape (input_name);
    std::vector<int> output_shape = input_shape; //输入与输出相等

    cout << "same out shape: " ;
    for (int i=0; i< output_shape.size(); ++i)
    {
        cout << output_shape[i] << " " ;
    }
    cout << endl;

    return output_shape;
}


//reshape  推理
std::vector<int> shapeInference::shape_inference_reshape(int node_index)
{
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    
    std::string reshape_name = m_model_proto.graph().node(node_index).input(1);
    std::vector<int> output_shape;
    
    for (int i=0; i < m_model_proto.graph().initializer_size(); ++i)
    {
        if (m_model_proto.graph().initializer(i).name() == reshape_name)
        {
            for (int j=0; j < m_model_proto.graph().initializer(i).int64_data_size(); ++j)
            {
                // cout << m_model_proto.graph().initializer(i).int64_data(j) << " ";
                output_shape.push_back(m_model_proto.graph().initializer(i).int64_data(j));
            }
            // cout << endl;
        }
    }

    cout  << "out shape: " ;
    for (int i=0; i< output_shape.size(); ++i)
    {
        cout << output_shape[i] << " " ;
    }
    cout << endl;
    return output_shape;
}

// add 等推理
std::vector<int> shapeInference::shape_inference_broadcast(int node_index)
{
    //Add 输入为A B, 矩阵相加，允许broadcasting
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    std::string a_name = m_model_proto.graph().node(node_index).input(0);
    std::string b_name = m_model_proto.graph().node(node_index).input(1);
    std::vector<int> a_shape = get_shape (a_name);
    std::vector<int> b_shape = get_shape (b_name);

    std::vector<int> output_shape;
    get_broadcast_shape(a_shape, b_shape, output_shape);

    cout << "broadcast out shape: " ;
    for (int i=0; i< output_shape.size(); ++i)
    {
        cout << output_shape[i] << " " ;
    }
    cout << endl;

    return output_shape;   
}

//matmul 推理
std::vector<int> shapeInference::shape_inference_matmul(int node_index)
{
     //Add 输入为A B, 矩阵相加，允许broadcasting
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    std::string a_name = m_model_proto.graph().node(node_index).input(0);
    std::string b_name = m_model_proto.graph().node(node_index).input(1);
    std::vector<int> a_shape = get_shape (a_name);
    std::vector<int> b_shape = get_shape (b_name);
    std::vector<int> output_shape;

    //必须维度相同，其他可乘
    int len_shape = a_shape.size();
    if ((a_shape.size()  != b_shape.size()) ||  (a_shape[len_shape-1] != b_shape[len_shape-2]))
    {
        cout << "error. shape canot natmul." << endl;
        return output_shape; 
    }

    for (int i=0; i < a_shape.size() -1; ++i)
    {
        output_shape.push_back(a_shape[i]);
    }
    output_shape.push_back(b_shape[len_shape-1]);

    return output_shape;   
}

//gemm 推理
std::vector<int> shapeInference::shape_inference_gemm(int node_index)
{
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    
    std::string a_name = m_model_proto.graph().node(node_index).input(0);
    std::string b_name = m_model_proto.graph().node(node_index).input(1);
    std::string c_name = m_model_proto.graph().node(node_index).input(2);
    std::vector<int> a_shape = get_shape (a_name);
    std::vector<int> b_shape = get_shape (b_name);

    int trans_a = 0;
    int trans_b = 0;
    for (int i=0; i< m_model_proto.graph().node(node_index).attribute_size(); ++i)
    {
        if (m_model_proto.graph().node(node_index).attribute(i).name() == "transA")
        {
            trans_a = m_model_proto.graph().node(node_index).attribute(i).i(); //todo: only support same stride
            cout << "transA: " <<  trans_a << endl;
        }
        else if (m_model_proto.graph().node(node_index).attribute(i).name() == "transB")
        {
            trans_b = m_model_proto.graph().node(node_index).attribute(i).i(); //todo: only support same kernel_shape
            cout << "transB: " <<  trans_b << endl;
        }
    }

    std::vector<int> a_shape_temp;
    std::vector<int> b_shape_temp;
    if (trans_a == 1)
    {
        for (int i=a_shape.size()-1; i>=0; --i)
        {
            a_shape_temp.push_back(a_shape[i]);
        }
    }
    else
    {
        a_shape_temp = a_shape;
    }

    if (trans_b == 1)
    {
        for (int i=b_shape.size()-1; i>=0; --i)
        {
            b_shape_temp.push_back(b_shape[i]);
        }
    }
    else
    {
        b_shape_temp = b_shape;
    }


    std::vector<int> output_shape;

    //必须维度相同，其他可乘
    int len_shape = a_shape_temp.size();
    if ((a_shape_temp.size()  != b_shape_temp.size()) ||  (a_shape_temp[len_shape-1] != b_shape_temp[len_shape-2]))
    {
        cout << "error. shape canot natmul." << endl;
        return output_shape; 
    }

    for (int i=0; i < a_shape_temp.size() -1; ++i)
    {
        output_shape.push_back(a_shape_temp[i]);
    }
    output_shape.push_back(b_shape_temp[len_shape-1]);

    return output_shape;   
}


// globalpooling 推理
std::vector<int> shapeInference::shape_inference_globalPooling(int node_index)
{
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    
    std::string input_name = m_model_proto.graph().node(node_index).input(0);
    std::vector<int> input_shape = get_shape (input_name);

    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    std::vector<int> output_shape;
    output_shape.push_back(N);
    output_shape.push_back(C);
    output_shape.push_back(1);
    output_shape.push_back(1);

    return output_shape;
}


std::vector<int> shapeInference::shape_inference_unsqueeze(int node_index)
{
    printf("-----\nshape inference: %s %s\n", m_model_proto.graph().node(node_index).op_type().c_str(), 
        m_model_proto.graph().node(node_index).name().c_str());
    
    std::string input_name = m_model_proto.graph().node(node_index).input(0);
    std::vector<int> input_shape = get_shape (input_name);
    std::vector<int> outpt_shape;

    std::vector<int> axes;
    for (int i=0; i< m_model_proto.graph().node(node_index).attribute_size(); ++i)
    {
        if (m_model_proto.graph().node(node_index).attribute(i).name() == "axes")
        {
            for (int j=0; j < m_model_proto.graph().node(node_index).attribute(i).ints_size(); ++j)
            {
                axes.push_back(m_model_proto.graph().node(node_index).attribute(i).ints(j));
            }
        }
    }

    // todo

    return outpt_shape;
}

// 返回  -1 失败   0 成功
int shapeInference::node_inference(int node_index)
{
    //如果上一层的形状未定义，返回推理失败
    for (int i=0; i<m_model_proto.graph().node(node_index).input_size();++i)
    {
        std::string prev_output_name = m_model_proto.graph().node(node_index).input(i);
        if (find_shape (prev_output_name) < 0)
        {
            return -1;
        }
    }

    //如果已经推理过，有output，直接返回
    if (find_shape (m_model_proto.graph().node(node_index).output(0)) >= 0)
    {
        return 0;
    }

    std::string op_type = m_model_proto.graph().node(node_index).op_type();
    std::vector<int> output_shape;
    if (op_type == "Conv" || op_type == "MaxPool" || op_type == "AveragePool")
    {
        output_shape = shape_inference_conv(node_index); 
    }
    else if (op_type == "Add" || op_type == "Sum")
    {
        output_shape = shape_inference_broadcast(node_index);
    }
    else if (op_type == "Relu" || op_type == "LRN" || op_type == "BatchNormalization" ||  op_type == "Softmax")
    {
        output_shape = shape_inference_same(node_index);
    }
    else if (op_type == "Reshape")
    {
        output_shape = shape_inference_reshape(node_index);
    }
    else if (op_type == "MatMul")
    {
        output_shape = shape_inference_matmul(node_index);
    }
    else if (op_type == "Gemm")
    {
        output_shape = shape_inference_gemm(node_index);
    }
    else if (op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool")
    {
        output_shape = shape_inference_globalPooling(node_index);
    }
    // else if (op_type == "Unsqueeze")
    // {
    //     output_shape = shape_inference_unsqueeze(node_index);
    // }
    else
    {
        cout << "error. " << op_type << " not support inference." << endl;
        return -1;
    }


    //推理结果放入shape_map中
    std::string output_name = m_model_proto.graph().node(node_index).output(0);
    m_shape_map.insert(std::pair<std::string,std::vector<int>>(output_name, output_shape));


    //推理将结果放入value_info中
    add_inference_shape(output_name, output_shape, m_input_type);     // type  FLOAT = 1;  
    // cout << "node shaped: " << m_model_proto.graph().node(node_index).name() << endl;

    return 0;
}




onnx::ModelProto shapeInference::inference()
{
    // 检查输入是否为固定形状，动态输入无法推理
    std::string input_name = m_model_proto.graph().input(0).name();
    cout << input_name  << ": " ;
    bool ok_flag = true;
    for (int i=0; i< m_model_proto.graph().input(0).type().tensor_type().shape().dim_size(); ++i)
    {
        int vaule = m_model_proto.graph().input(0).type().tensor_type().shape().dim(i).dim_value();
        cout <<  vaule << " " ;
        // cout <<  m_model_proto.graph().input(0).type().tensor_type().shape().dim(i).dim_param() << endl;
        if (vaule == 0)
        {
            ok_flag = false;
        }
    }
    cout << endl;
    if (!ok_flag)
    {
        cout <<  "error! input shape unkown, can not shape inference!" << endl ;
        return m_model_proto;
    }

    int k = m_model_proto.graph().node_size();
    while (k--)
    {
        for (int i=0; i< m_model_proto.graph().node_size(); ++i)
        {
            //推理当前node的output形状
            node_inference(i);
        }
    }

    return m_model_proto;
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
        std::fstream input(argv[1], ios::in | ios::binary);
        if (!input) 
        {
            std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
        } 
        else if (!model_proto.ParseFromIstream(&input)) 
        {
            std::cerr << "Failed to parse onnx model." << std::endl;
            return -1;
        }
    }

    onnx::ModelProto new_model_proto;
    shapeInference my_shape(model_proto);
    new_model_proto = my_shape.inference();

    {
        // Write the new model back to disk.
        std::string output_file_name  = std::string(argv[2]) + ".onnx";
        fstream output(output_file_name, ios::out | ios::trunc | ios::binary);
        if (!new_model_proto.SerializeToOstream(&output)) 
        {
            std::cerr << "Failed to write address book." << endl;
            return -1;
        }
    }


    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

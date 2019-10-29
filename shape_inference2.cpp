#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 

#define ONNX_API ONNX_IMPORT
#define ONNX_NAMESPACE onnx  //注意命名 否则编译失败

#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"


using namespace std;

int main(int argc, char const *argv[]) 
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " input_onnx_file  " << "output_name" <<'\n';
		return -1;
	}

	onnx::ModelProto model_in;

	{
		//读取model
		std::fstream input(argv[1], ios::in | ios::binary);
		if (!input) {
			std::cout << argv[1] << ": file not found." << '\n';
			return -1;
		} else if (!model_in.ParseFromIstream(&input)) {
			std::cerr << "Failed to parse onnx model." << std::endl;
			return -1;
		}
	}

	// // 进行shape_inference
	onnx::ModelProto model_out = model_in;
	onnx::shape_inference::InferShapes(model_out);
	
	{
		// 保存optimize后的model
		fstream output(argv[2], ios::out | ios::trunc | ios::binary);
		if (!model_out.SerializeToOstream(&output)) 
		{
		  cerr << "Failed to create onnx model: " <<  argv[2] <<endl;
		  return -1;
		}
		std::cout << "save model " << argv[2] <<endl;
	}
		
	google::protobuf::ShutdownProtobufLibrary();
	
	return 0;

}



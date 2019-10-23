import onnx
from onnx import optimizer
import sys

# Preprocessing: load the model to be optimized.
model_path = sys.argv[1]
original_model = onnx.load(model_path)
# print('The model before optimization:\n{}'.format(original_model))

all_passes = optimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print(p)
print()

# Pick one pass as example
#passes = ['fuse_add_bias_into_conv']
passes = ['eliminate_deadend']

# Apply the optimization on the original model
optimized_model = optimizer.optimize(original_model, passes)
# print('The model after optimization:\n{}'.format(optimized_model))
onnx.save(optimized_model, './opt1.onnx')


# One can also apply the default passes on the (serialized) model
# Check the default passes here: https://github.com/onnx/onnx/blob/master/onnx/optimizer.py#L43
optimized_model = optimizer.optimize(original_model)
onnx.save(optimized_model, './opt2.onnx')

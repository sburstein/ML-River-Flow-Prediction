from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
#model_dir = "tf_wx_model"
#checkpoint_path = os.path.join(model_dir, "model.ckpt")

# modify checkpoint_path based on where tf_wx_model/model.ckpt-40000 is stored on your computer
checkpoint_path = "/Users/ScottBurstein/Desktop/tf_wx_model/model.ckpt-40000"
#model.ckpt-40000.data-00000-of-00001

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name=all, all_tensors=True)

# List contents of v0 tensor.
# Example output: tensor_name:  v0 [[[[  9.27958265e-02   7.40226209e-02   4.52989563e-02   3.15700471e-02
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v0', all_tensors=True)

# List contents of v1 tensor.
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='v1', all_tensors=True)
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include "../include/libtorch.h"
#include <petsc.h>

torch::jit::script::Module model;
torch::DeviceType device = torch::kXPU;

PetscErrorCode LoadModel_LibTorch(const char *model_path) {
  PetscFunctionBeginUser;

  PetscCallCXX(model = torch::jit::load(model_path));

  PetscCallCXX(model.to(torch::Device(device)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Load and run model
PetscErrorCode ModelInference_LibTorch(Vec DD_Inputs_loc, Vec DD_Outputs_loc) {

  PetscMemType input_mem_type;
  PetscInt input_size, num_nodes;
  const PetscScalar *dd_inputs_ptr;
  PetscFunctionBeginUser;

  PetscCall(VecGetLocalSize(DD_Inputs_loc, &input_size));
  num_nodes = input_size / 6;
  PetscCall(VecGetArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr, &input_mem_type));

  torch::TensorOptions options;
  torch::Tensor gpu_tensor;

  PetscCallCXX(options = torch::TensorOptions()
                          .dtype(torch::kFloat64)
                          .device(device));
  PetscCallCXX(gpu_tensor = torch::from_blob((void *)dd_inputs_ptr, {num_nodes, 6}, options));

  // PetscCallCXX(gpu_tensor = torch::rand({512,6}, options));

  // Run model
  torch::Tensor output;
  PetscCallCXX(output = model.forward({gpu_tensor}).toTensor());
  PetscCall(VecRestoreArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr));

  {
    PetscMemType  output_mem_type;
    PetscInt      output_size, num_nodes;
    PetscScalar  *dd_outputs_ptr;
    torch::Tensor DD_Outputs_tensor;

    PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
    num_nodes = input_size / 6;
    PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));

    PetscCallCXX(DD_Outputs_tensor = torch::from_blob((void *)dd_outputs_ptr, {num_nodes, 6}, options));

    PetscCallCXX(DD_Outputs_tensor.copy_(output));

    PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopyTest(Vec DD_Outputs_loc) {
  PetscMemType output_mem_type;
  PetscInt output_size, num_nodes;
  PetscScalar *dd_outputs_ptr;
  PetscFunctionBeginUser;

  PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
  PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));

  auto dims = torch::IntArrayRef{output_size};
  auto options = torch::TensorOptions()
                          .dtype(torch::kFloat64)
                          .device(device);
  torch::Tensor gpu_tensor = torch::from_blob((void *)dd_outputs_ptr, dims, options);

  torch::Tensor ones = torch::ones_like(gpu_tensor);

  gpu_tensor.copy_(ones);

  PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Load a model and return pointer to it
void upload_model(void** model_ptr) {

  std::string model_name = "./resnet50_jit.pt";
  try {
    //std::cout << "loading the model " << model_name << "\n" << std::flush;
    auto model_tmp = torch::jit::load(model_name);
    torch::jit::Module* model = new torch::jit::Module(model_tmp);
    model->to(torch::Device(torch::kXPU));
    std::cout << "loaded the model to GPU" << std::flush;
    *model_ptr = NULL;
    *model_ptr = reinterpret_cast<void*>(model);
    //printf("%p\n", model);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << std::flush;
  }

}

// Execute a model on data passed through the function
//void torch_inf(void *model_ptr, void *inputs, void *outputs) {
void torch_inf(void *model_ptr) {

  // Convert input array to Torch tensor
  auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kXPU);
  torch::Tensor input_tensor = torch::rand({512,6}, options);
  //torch::Tensor input_tensor = torch::from_blob(inputs, {*batch,*channels,*pixels,*pixels}, torch::dtype(torch::kFloat32));
  //input_tensor = input_tensor.to(torch::Device(torch::kCUDA, *myGPU));
  //std::cout << "created the input vector\n" << std::flush;

  // Convert Tensor to vector
  //std::vector<torch::jit::IValue> model_inputs;
  //model_inputs.push_back(input_tensor);
  //std::cout << "prepared input vector for inference\n";

  // Perform inference
  torch::jit::Module* module = reinterpret_cast<torch::jit::Module*>(model_ptr);
  //torch::Tensor output_tensor = module->forward(model_inputs).toTensor();
  torch::Tensor output_tensor = module->forward({input_tensor}).toTensor();
  std::cout << "performed inference\n" << std::flush;

  // Extract predictions
  //outputs = (void *) output_tensor.data_ptr<float>();

}

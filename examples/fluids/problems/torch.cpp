#include <iostream>
#include <petsc.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "../include/libtorch.h"

torch::jit::script::Module model;
// torch::DeviceType          device;
torch::DeviceType          device = torch::kXPU;
// torch::DeviceType          device = torch::kCPU;

torch::DeviceType Enum2DeviceType(TorchDeviceType device_enum) {
  switch (device_enum) {
    case TORCH_DEVICE_CPU:
      return torch::kCPU;
    case TORCH_DEVICE_XPU:
      return torch::kXPU;
    case TORCH_DEVICE_CUDA:
      return torch::kCUDA;
  }
}

PetscErrorCode LoadModel_LibTorch(const char *model_path, TorchDeviceType device_enum) {
  PetscFunctionBeginUser;

  // PetscCallCXX(device = torch::Device(device_str));
  // PetscCallCXX(device = Enum2DeviceType(device_enum));
  // PetscCallCXX(device = torch::kXPU);

  PetscCallCXX(model = torch::jit::load(model_path));
  PetscCallCXX(model.to(torch::Device(device)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Load and run model
PetscErrorCode ModelInference_LibTorch_Host(Vec DD_Inputs_loc, Vec DD_Outputs_loc) {
  torch::Tensor        gpu_tensor, output;

  PetscFunctionBeginUser;
  // torch::NoGradGuard no_grad; // equivalent to "with torch.no_grad():" in PyTorch
  {
    PetscInt           input_size, num_nodes;
    const PetscScalar *dd_inputs_ptr;
    torch::TensorOptions options;

    PetscCall(VecGetLocalSize(DD_Inputs_loc, &input_size));
    num_nodes = input_size / 6;
    PetscCall(VecGetArrayRead(DD_Inputs_loc, &dd_inputs_ptr));

    PetscCallCXX(options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    PetscCallCXX(gpu_tensor = torch::from_blob((void *)dd_inputs_ptr, {num_nodes, 6}, options));
    PetscCall(VecRestoreArrayRead(DD_Inputs_loc, &dd_inputs_ptr));
  }

  // Run model
  PetscCallCXX(output = model.forward({gpu_tensor}).toTensor());

  {
    PetscInt     output_size;
    PetscScalar *dd_outputs_ptr;
    double       *tensor_output = (double *)output.contiguous().to(torch::kCPU).data_ptr();

    PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
    PetscCall(VecGetArray(DD_Outputs_loc, &dd_outputs_ptr));

    PetscCall(PetscArraycpy(dd_outputs_ptr, tensor_output, output_size));

    PetscCall(VecRestoreArray(DD_Outputs_loc, &dd_outputs_ptr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Load and run model
PetscErrorCode ModelInference_LibTorch(Vec DD_Inputs_loc, Vec DD_Outputs_loc) {
  torch::Tensor        gpu_tensor, output;
  torch::TensorOptions options;
  const PetscInt num_input_comps = 6, num_output_comps = 6;
  PetscFunctionBeginUser;
  // torch::NoGradGuard no_grad; // equivalent to "with torch.no_grad():" in PyTorch
  {
    PetscMemType       input_mem_type;
    PetscInt           input_size, num_nodes;
    const PetscScalar *dd_inputs_ptr;

    PetscCall(VecGetLocalSize(DD_Inputs_loc, &input_size));
    num_nodes = input_size / num_input_comps;
    PetscCall(VecGetArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr, &input_mem_type));

    PetscCallCXX(options = torch::TensorOptions().dtype(torch::kFloat64).device(device));
    if (device == torch::kXPU) {  // XPU requires device-to-host-to-device transfer
      PetscCallCXX(gpu_tensor = at::from_blob((void *)dd_inputs_ptr, {num_nodes, num_input_comps}, {num_input_comps, 1}, nullptr, options, device).to(device));
    } else {
      PetscCallCXX(gpu_tensor = torch::from_blob((void *)dd_inputs_ptr, {num_nodes, num_input_comps}, options));
    }
    PetscCall(VecRestoreArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr));
  }

  // Run model
  PetscCallCXX(output = model.forward({gpu_tensor}).toTensor());

  if (device == torch::kXPU) {  // XPU requires device-to-host-to-device transfer
    PetscInt     output_size;
    PetscScalar *dd_outputs_ptr;
    double       *tensor_output;

    PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
    PetscCall(VecGetArray(DD_Outputs_loc, &dd_outputs_ptr));
    PetscCallCXX(tensor_output = (double *)output.contiguous().to(torch::kCPU).data_ptr());

    PetscCall(PetscArraycpy(dd_outputs_ptr, tensor_output, output_size));

    PetscCall(VecRestoreArray(DD_Outputs_loc, &dd_outputs_ptr));
  } else {
    PetscMemType  output_mem_type;
    PetscInt      output_size, num_nodes;
    PetscScalar  *dd_outputs_ptr;
    torch::Tensor DD_Outputs_tensor;

    PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
    num_nodes = output_size / num_output_comps;
    PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));

    PetscCallCXX(DD_Outputs_tensor = torch::from_blob((void *)dd_outputs_ptr, {num_nodes, num_output_comps}, options));

    PetscCallCXX(DD_Outputs_tensor.copy_(output));

    PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CopyTest(Vec DD_Outputs_loc) {
  PetscMemType output_mem_type;
  PetscInt     output_size, num_nodes;
  PetscScalar *dd_outputs_ptr;
  PetscFunctionBeginUser;

  PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
  PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));

  auto          dims       = torch::IntArrayRef{output_size};
  auto          options    = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  torch::Tensor gpu_tensor = torch::from_blob((void *)dd_outputs_ptr, dims, options);

  torch::Tensor ones = torch::ones_like(gpu_tensor);

  gpu_tensor.copy_(ones);

  PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

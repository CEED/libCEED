// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <log_events.h>
#include <petsc.h>
#include <sgs_model_torch.h>
#include <torch/script.h>
#include <torch/torch.h>

torch::jit::script::Module model;
torch::DeviceType          device_model;

static PetscErrorCode EnumToDeviceType(TorchDeviceType device_enum, torch::DeviceType *device_type) {
  PetscFunctionBeginUser;
  switch (device_enum) {
    case TORCH_DEVICE_CPU:
      *device_type = torch::kCPU;
      break;
    case TORCH_DEVICE_XPU:
      *device_type = torch::kXPU;
      break;
    case TORCH_DEVICE_CUDA:
      *device_type = torch::kCUDA;
      break;
    case TORCH_DEVICE_HIP:
      *device_type = torch::kHIP;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "TorchDeviceType %d not supported by PyTorch inference", device_enum);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscMemTypeToDeviceType(PetscMemType mem_type, torch::DeviceType *device_type) {
  PetscFunctionBeginUser;
  switch (mem_type) {
    case PETSC_MEMTYPE_HOST:
      *device_type = torch::kCPU;
      break;
    case PETSC_MEMTYPE_SYCL:
      *device_type = torch::kXPU;
      break;
    case PETSC_MEMTYPE_CUDA:
      *device_type = torch::kCUDA;
      break;
    case PETSC_MEMTYPE_HIP:
      *device_type = torch::kHIP;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PetscMemType %s not supported by PyTorch inference", PetscMemTypeToString(mem_type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LoadModel_Torch(const char *model_path, TorchDeviceType device_enum) {
  PetscFunctionBeginUser;
  PetscCall(EnumToDeviceType(device_enum, &device_model));

  PetscCallCXX(model = torch::jit::load(model_path));
  PetscCallCXX(model.to(torch::Device(device_model)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Load and run model
PetscErrorCode ModelInference_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc) {
  torch::Tensor  input_tensor, output_tensor;
  const PetscInt num_input_comps = 6, num_output_comps = 6;
  PetscBool      debug_tensor_output = PETSC_FALSE;

  PetscFunctionBeginUser;
  // torch::NoGradGuard no_grad; // equivalent to "with torch.no_grad():" in PyTorch
  PetscCall(PetscLogEventBegin(FLUIDS_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  {  // Transfer DD_Inputs_loc into input_tensor
    PetscMemType         input_mem_type;
    PetscInt             input_size, num_nodes;
    const PetscScalar   *dd_inputs_ptr;
    torch::DeviceType    dd_input_device;
    torch::TensorOptions options;

    PetscCall(VecGetLocalSize(DD_Inputs_loc, &input_size));
    num_nodes = input_size / num_input_comps;
    PetscCall(VecGetArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr, &input_mem_type));
    PetscCall(PetscMemTypeToDeviceType(input_mem_type, &dd_input_device));

    PetscCallCXX(options = torch::TensorOptions().dtype(torch::kFloat64).device(dd_input_device));
    if (dd_input_device == torch::kXPU) {  // XPU requires device-to-host-to-device transfer
      PetscCallCXX(input_tensor =
                       at::from_blob((void *)dd_inputs_ptr, {num_nodes, num_input_comps}, {num_input_comps, 1}, nullptr, options, dd_input_device)
                           .to(device_model));
    } else {
      PetscCallCXX(input_tensor = torch::from_blob((void *)dd_inputs_ptr, {num_nodes, num_input_comps}, options));
    }
    if (debug_tensor_output) {
      double *input_tensor_ptr;

      PetscCall(VecGetLocalSize(DD_Inputs_loc, &input_size));
      PetscCallCXX(input_tensor_ptr = (double *)input_tensor.contiguous().to(torch::kCPU).data_ptr());
      printf("Input_Tensor_Pointer:\n");
      for (PetscInt i = 0; i < input_size; i++) {
        printf("%f\n", input_tensor_ptr[i]);
      }
    }
    PetscCall(VecRestoreArrayReadAndMemType(DD_Inputs_loc, &dd_inputs_ptr));
  }
  PetscCall(PetscLogEventEnd(FLUIDS_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));

  // Run model
  PetscCall(PetscLogEventBegin(FLUIDS_SgsModelDDInference, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(output_tensor = model.forward({input_tensor}).toTensor());
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(FLUIDS_SgsModelDDInference, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));

  PetscCall(PetscLogEventBegin(FLUIDS_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  {  // Transfer output_tensor to DD_Outputs_loc
    torch::DeviceType    dd_output_device;
    torch::TensorOptions options;
    PetscInt             output_size;
    PetscScalar         *dd_outputs_ptr;
    PetscMemType         output_mem_type;

    {  // Get DeviceType of DD_Outputs_loc
      PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));
      PetscCall(PetscMemTypeToDeviceType(output_mem_type, &dd_output_device));
      PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
    }

    if (dd_output_device == torch::kXPU) {  // XPU requires device-to-host-to-device transfer
      double *output_tensor_ptr;

      PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
      PetscCall(VecGetArray(DD_Outputs_loc, &dd_outputs_ptr));
      PetscCallCXX(output_tensor_ptr = (double *)output_tensor.contiguous().to(torch::kCPU).data_ptr());
      if (debug_tensor_output) {
        printf("Output_Tensor_Pointer:\n");
        for (PetscInt i = 0; i < output_size; i++) {
          printf("%f\n", output_tensor_ptr[i]);
        }
      }
      PetscCall(PetscArraycpy(dd_outputs_ptr, output_tensor_ptr, output_size));
      PetscCall(VecRestoreArray(DD_Outputs_loc, &dd_outputs_ptr));
    } else {
      PetscInt      num_nodes;
      torch::Tensor DD_Outputs_tensor;

      PetscCall(VecGetLocalSize(DD_Outputs_loc, &output_size));
      num_nodes = output_size / num_output_comps;
      PetscCall(VecGetArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr, &output_mem_type));
      PetscCallCXX(options = torch::TensorOptions().dtype(torch::kFloat64).device(dd_output_device));
      PetscCallCXX(DD_Outputs_tensor = torch::from_blob((void *)dd_outputs_ptr, {num_nodes, num_output_comps}, options));
      PetscCallCXX(DD_Outputs_tensor.copy_(output_tensor));
      PetscCall(VecRestoreArrayAndMemType(DD_Outputs_loc, &dd_outputs_ptr));
    }
  }
  PetscCall(PetscLogEventEnd(FLUIDS_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

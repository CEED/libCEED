#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

extern "C" {
  int create_tensor();
  int load_and_run();
  void upload_model(void** model_ptr);
  void torch_inf(void* model_ptr);
}

// Simple function to create a tensor, offload it and perform matrix multiply
int create_tensor() {

  // Create tensors on CPU
  torch::Tensor atensor = torch::rand({64, 64});
  std::cout << "Tensor created on device " << atensor.device().type() << std::endl;
  //std::cout << "Tensor is : " << tensor << "\n" << std::endl;
  torch::Tensor btensor = torch::rand({64, 64});

  // Offload tensors to GPU
  atensor = atensor.to(torch::Device(torch::kXPU));
  std::cout << "Tensor offloaded to device " << atensor.device().type() << std::endl;
  btensor = btensor.to(torch::Device(torch::kXPU));

  // Perform matrix multiplication
  torch::Tensor ctensor = torch::matmul(atensor,btensor);
  std::cout << "Performed Torch matrix multiply" << std::endl;
  std::cout << "Output tensor located on device " << ctensor.device().type() << std::endl;

  return 0;
}

// Load and run model
int load_and_run() {

  // Upload model
  torch::jit::script::Module model;
  try {
        model = torch::jit::load("NNmodel_jit_inf.pt");
        std::cout << "Loaded the model\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    // Offload model to GPU
    model.to(torch::Device(torch::kXPU));
    std::cout << "Model offloaded to GPU\n";
  
  // Create input tensor
  auto options = torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(torch::kXPU);
  torch::Tensor input_tensor = torch::rand({512,6}, options);
  assert(input_tensor.dtype() == torch::kFloat32);
  assert(input_tensor.device().type() == torch::kXPU);
  std::cout << "Created the input tesor on GPU\n";

  // Run model
  torch::Tensor output = model.forward({input_tensor}).toTensor();
  std::cout << "Performed inference\n";

  return 0;
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

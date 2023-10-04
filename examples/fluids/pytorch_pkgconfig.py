from pathlib import Path
import torch
import torch.utils.cpp_extension as C
import torch.utils as tutils
import re

build_dir = Path('./build')
pkgconfig_path = build_dir / 'libtorch.pc'

variables = {}
keywords = {}


def add_variable(file, variable, value):
    file.write(f"{variable}={value}\n")


def add_keyword(file, key, value):
    file.write(f"{key}: {value}\n")


variables['prefix'] = Path(C.library_paths()[0]).parent.as_posix()

keywords['Name'] = 'libTorch'
keywords['Description'] = 'Custom made PC for PyTorch'
keywords['Version'] = torch.__version__

keywords['Cflags'] = ''
for include_path in C.include_paths():
    keywords['Cflags'] += f'-I{include_path} '

# Need to search the CMake file to see whether the library was compiled with the CXX11 ABI standard
regex_ABI = re.compile(r'"(\S*GLIBCXX_USE_CXX11_ABI\S*)"')
torchCMakePath = Path(tutils.cmake_prefix_path) / 'Torch/TorchConfig.cmake'
abi_flag = ''
with torchCMakePath.open('r') as f:
    for line in f:
        regex_result = regex_ABI.search(line)
        if regex_result:
            abi_flag = regex_result[1]

keywords['Cflags'] += abi_flag

keywords['Libs'] = ''
for lib_path in C.library_paths():
    keywords['Libs'] += f'-L{lib_path} '
keywords['Libs'] += '-lc10 -ltorch_cpu '
if torch.cuda.is_available():
    keywords['Libs'] += '-lc10_cuda -ltorch_cuda '
    # Need to force linking with libtorch_cuda.so, so find path and specify linking flag to force it
    # This flag might be of limited portability
    for lib_path in C.library_paths():
        torch_cuda_path = Path(lib_path) / 'libtorch_cuda.so'
        if torch_cuda_path.exists():
            variables['torch_cuda_path'] = torch_cuda_path.as_posix()
            keywords['Libs'] += f'-Wl,--no-as-needed,"{torch_cuda_path.as_posix()}" '
keywords['Libs'] += '-ltorch '
keywords['Libs.private'] = ''

with pkgconfig_path.open('w') as file:
    for variable, value in variables.items():
        add_variable(file, variable, value)

    file.write('\n')

    for keyword, value in keywords.items():
        add_keyword(file, keyword, value)

print(pkgconfig_path.absolute())

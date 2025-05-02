from setuptools.command.build_ext import build_ext
import sys
import os

class BuildExt(build_ext):
    def build_extensions(self):
        # Avoid warning about -Wstrict-prototypes
        if sys.platform == 'darwin':
            if '-Wstrict-prototypes' in self.compiler.compiler_so:
                self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()

# Define the extension module
ext_modules = [
    Extension(
        "ex3_volume_qfunctions",
        sources=["ex3_volume_qfunctions.c"],
        include_dirs=[os.path.join("..", "..", "include")],
        libraries=["ceed"],
        library_dirs=[os.path.join("..", "..", "lib")],
        extra_compile_args=["-fPIC"],
    )
]

setup(
    name="ex3_volume_qfunctions",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
) 

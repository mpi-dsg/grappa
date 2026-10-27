import os
import subprocess

from setuptools import setup
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

from t10n import __version__

class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class cmake_build_ext(build_ext):

    def configure(self, ext: CMakeExtension) -> None:

        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        # where .so files will be written, should be the same for all extensions
        # that use the same CMakeLists.txt.
        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(outdir),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}'.format(self.build_temp),
        ]

        cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            target = ext.name
            if target.startswith("t10n."):
                target=target[5:]
            build_args = [
                '--build', '.', '-j16', '--target', target
            ]

            subprocess.check_call(['cmake', *build_args], cwd=self.build_temp)

ext_modules = []
ext_modules.append(CMakeExtension(name="t10n._C")) # t10n/_C.xxx.so

setup(
    name="t10n",
    version=__version__,
    author="foo",
    author_email="foo@bar",
    url="foo.bar",
    description="foobar",
    long_description="foobar",
    packages=find_packages(exclude=("csrc", "cmake", "docker", "exp", "scripts")),
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": cmake_build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

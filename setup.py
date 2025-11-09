from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from Cython.Build import cythonize
import os
import zipfile
import tempfile


class BinaryWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        super().run()
        
        wheel_path = None
        for file in os.listdir(self.dist_dir):
            if file.endswith('.whl'):
                wheel_path = os.path.join(self.dist_dir, file)
                break
        
        if wheel_path and os.path.exists(wheel_path):
            print(f" Post-processing wheel: {wheel_path}")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                cleaned = 0
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if (f.endswith(".py") or f.endswith(".c")) and f != "__init__.py":
                            filepath = os.path.join(root, f)
                            try:
                                os.remove(filepath)
                                cleaned += 1
                                print(f"  Removed: {os.path.relpath(filepath, tmpdir)}")
                            except Exception as e:
                                print(f"  Warning: couldn't remove {filepath}: {e}")
                
                print(f"🧹 Removed {cleaned} .py/.c files from wheel")
                
                os.remove(wheel_path)
                with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for root, _, files in os.walk(tmpdir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, tmpdir)
                            zip_ref.write(file_path, arcname)
                
                print(f"Repackaged wheel: {wheel_path}")


extensions = cythonize(
    ["fall_package/**/*.py"],
    language_level=3,
    exclude=["**/__init__.py"],
    compiler_directives={"always_allow_keywords": True}
)

setup(
    name="fall_package",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    package_data={
        "fall_package": [
            "weights/*.enc",
            "*.so",
            "*.pyd",
            "__init__.py"
        ]
    },
    cmdclass={
        "bdist_wheel": BinaryWheel,  
    },
)
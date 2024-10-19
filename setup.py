# Run `pip install -e .` in the root dir to use the packages
from setuptools import find_packages, setup

# There are modules inside LucidDreamer that need to be found as well
setup(
    name="PromptCtrl",
    version="1.0",
    packages=find_packages(),
    # packages=["ctrl_utils", "arguments", "gaussian_renderer", "scene"],
    # package_dir={
    #     "ctrl_utils": ".",
    #     "arguments": "./ctrl_3d/LucidDreamer/arguments",
    #     "gaussian_renderer": "./ctrl_3d/LucidDreamer/gaussian_renderer",
    #     "scene": "./ctrl_3d/LucidDreamer/scene",
    # }
)

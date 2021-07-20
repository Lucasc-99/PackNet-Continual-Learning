import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["packnet*"])
# packages = ["packnet"] 
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="packnet",
    version="0.0.1",
    author="Lucas Cecchi",
    author_email="lucascecchi@gmail.com",
    description="Generic PackNet method to be added to Sequoia.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lucasc-99/PackNet-Continual-Learning",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "Method": [
            "packnet = packnet.packnet_method:PackNetMethod",
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        "tensorboardx",
        "simple_parsing>=0.0.15.post1",
    ],
)

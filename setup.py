# File: setup.py
from setuptools import find_packages, setup

setup(
    name="SEdist",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

# Use with: python -m pip install -e . 
# allows files to remain editable after installation.
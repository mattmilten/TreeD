from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="treed",
    version="0.0.2",
    author="Matthias Miltenberger",
    author_email="matthias.miltenberger@gmail.com",
    description="3D Visualization of Branch-and-Cut Trees using PySCIPOpt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattmilten/TreeD",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pyscipopt", "scikit-learn", "pandas", "plotly", "networkx"],
    python_requires=">=3.6",
    scripts=['bin/treed'],
)

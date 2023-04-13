import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lars_evaluator",
    version="0.1.0",
    author="Lojze Žust",
    author_email="lojze.zust@fri.uni-lj.si",
    description="Evaluation toolkit for the LaRS dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lojzezust/lars_evaluator",
    packages=['lars_eval'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'panopticapi @ git+https://github.com/cocodataset/panopticapi.git',
        'Pillow',
        'numpy',
        'yacs',
        'tqdm',
        'opencv-python',
        'pandas'
    ]
)

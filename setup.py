from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="seg_cl_pipeline",
    version="2.2.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-pipeline=v2.runner:run_pipeline",
        ],
    },
    author="Moritz Schillinger",
    author_email="moritz.schillinger@fau.de",
    description="An image segmentation and analysis pipeline for MSOT research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="hhttps://github.com/Mo-Sc/OA-US-Analysis-Pipeline",
    python_requires=">=3.10",
)

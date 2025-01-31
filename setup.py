# -*- coding: utf-8 -*-
import setuptools
from setuptools import find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name="langchain-deepseek",
    version="0.0.3",
    description="langchain-deepseek",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="1755115828@qq.com",
    url="https://github.com/SyJarvis/langchain-deepseek",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain>=0.2.6"
    ],
    extras_require={
        "finetune": ["deepspeed", "flash-attn"]
    }
)
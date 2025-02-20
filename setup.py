from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name="asr-system",
    version="1.0.0",
    author="Ali Moridani",
    author_email="moridani@gmail.com",
    description="Advanced Multilingual Speech Recognition System",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/asr-system",
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-cov>=4.1.0',
            'black>=24.1.1',
            'mypy>=1.8.0',
            'flake8>=7.0.0',
        ],
        'gpu': [
            'torch>=2.2.1',
            'nvidia-cuda-runtime-cu12==12.1.105',
            'nvidia-cublas-cu12==12.1.3.1',
            'nvidia-cudnn-cu12==8.9.2.26',
        ],
    },
    entry_points={
        'console_scripts': [
            'asr-system=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['config/*.json', 'config/*.yaml'],
    },
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/moridani/asr-system/issues",
        "Documentation": "https://github.com/moridani/asr-system/wiki",
        "Source Code": "https://github.com/moridani/asr-system",
    },
)
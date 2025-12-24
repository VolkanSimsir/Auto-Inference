import os
from pathlib import Path

from setuptools import find_packages, setup

# Read the README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read version from package
version = "0.2.0"

# Core dependencies (minimal)
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "huggingface_hub>=0.16.0",
]

# Optional dependencies for different backends
EXTRAS_REQUIRE = {
    # Individual backends
    "transformers": [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
    ],
    "unsloth": [
        "unsloth>=2024.1",
    ],
    "vllm": [
        "vllm>=0.2.0",
    ],
    "llamacpp": [
        "llama-cpp-python>=0.2.0",
    ],
    # Convenience groups
    "all": [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "vllm>=0.2.0",
        "llama-cpp-python>=0.2.0",
    ],
    "all-gpu": [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "unsloth>=2024.1",
        "vllm>=0.2.0",
    ],
    "cpu": [
        "transformers>=4.35.0",
        "llama-cpp-python>=0.2.0",
    ],
    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "ruff>=0.1.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
}

setup(
    name="auto-inference",
    version=version,
    author="Kadir Nar",
    author_email="kadir.nar@example.com",
    description="A unified interface for LLM inference across multiple backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kadirnar/Auto-Inference",
    project_urls={
        "Bug Tracker": "https://github.com/kadirnar/Auto-Inference/issues",
        "Documentation": "https://github.com/kadirnar/Auto-Inference#readme",
        "Source Code": "https://github.com/kadirnar/Auto-Inference",
    },
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "llm",
        "inference",
        "transformers",
        "vllm",
        "unsloth",
        "llama-cpp",
        "language-model",
        "machine-learning",
        "deep-learning",
        "nlp",
        "huggingface",
    ],
    zip_safe=False,
)

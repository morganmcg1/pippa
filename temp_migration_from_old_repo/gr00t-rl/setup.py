"""Setup script for GR00T-RL package."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gr00t-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Reinforcement Learning for NVIDIA GR00T Robot Foundation Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/morganmcg1/pippa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.12.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "gputil>=1.4.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.261",
            "pre-commit>=3.2.0",
        ],
        "isaac": [
            "isaacsim-rl",  # Isaac Lab dependencies
            "omniverse-isaac-sim",
        ],
    },
    entry_points={
        "console_scripts": [
            "gr00t-train=scripts.train_ppo_v2:main",
            "gr00t-test=scripts.test_ppo_wandb:main",
        ],
    },
)
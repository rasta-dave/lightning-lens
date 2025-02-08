from setuptools import setup, find_packages

setup(
    name="lightninglens",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "networkx>=2.8.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "lightning-rpc>=0.3.0",
        "pyyaml>=6.0.0",
        "plotly>=5.13.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "jupyter>=1.0.0",
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-driven liquidity prediction and optimization for Lightning Network nodes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lightninglens",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)

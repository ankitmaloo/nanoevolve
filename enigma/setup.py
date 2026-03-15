from setuptools import setup, find_packages

setup(
    name="enigma",
    version="0.1.0",
    description="Universal program evolution engine",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.30.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "gemini": ["google-generativeai>=0.3.0"],
        "all": ["openai>=1.0.0", "google-generativeai>=0.3.0"],
    },
    entry_points={
        "console_scripts": [
            "enigma=enigma.cli:main",
        ],
    },
)

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="tibetan_text_metrics",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(["src/fast_lcs.pyx"]),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "python-Levenshtein>=0.12.0",
        "gensim>=4.0.0",
        "tqdm>=4.60.0",
        "Cython>=0.29.0",
    ],
    python_requires=">=3.8",
)
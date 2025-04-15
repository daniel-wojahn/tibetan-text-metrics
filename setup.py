import numpy
from setuptools import Extension, setup

setup(
    version="0.3.0",
    ext_modules=[
        Extension(
            "tibetan_text_metrics.fast_lcs",
            ["src/tibetan_text_metrics/fast_lcs.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "tibetan_text_metrics.fast_patterns",  # New fast_patterns module
            ["src/tibetan_text_metrics/fast_patterns.pyx"],  # Cython file path
            include_dirs=[numpy.get_include()],
        ),
    ],
    package_dir={"": "src"},
)

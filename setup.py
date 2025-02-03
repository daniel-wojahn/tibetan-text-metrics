import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "tibetan_text_metrics.fast_lcs",
            ["src/tibetan_text_metrics/fast_lcs.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ],
    package_dir={"": "src"},
)

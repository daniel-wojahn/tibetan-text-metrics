import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

# It's good practice to specify encoding for portability
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tibetan text metrics webapp",
    version="0.1.0",
    author="Daniel Wojahn / Tibetan Text Metrics",
    author_email="daniel.wojahn@outlook.de",
    description="Cython components for the Tibetan Text Metrics Webapp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel-wojahn/tibetan-text-metrics",
    ext_modules=cythonize(
        [
            Extension(
                "pipeline.fast_lcs",  # Module name to import: from pipeline.fast_lcs import ...
                ["pipeline/fast_lcs.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ],
        compiler_directives={'language_level' : "3"} # For Python 3 compatibility
    ),
    # Indicates that package data (like .pyx files) should be included if specified in MANIFEST.in
    # For simple cases like this, Cythonize usually handles it.
    include_package_data=True, 
    # Although this setup.py is in webapp, it's building modules for the 'pipeline' sub-package
    # We don't list packages here as this setup.py is just for the extension.
    # The main app will treat 'pipeline' as a regular package.
    zip_safe=False, # Cython extensions are generally not zip-safe
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20", # Ensure numpy is available for runtime if not just build time
    ],
    # setup_requires is deprecated, use pyproject.toml for build-system requirements
)

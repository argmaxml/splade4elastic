from distutils.core import setup

__package__= "splade4elastic"
__version__=""
with open(__package__+"/__init__.py", 'r') as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line)
            break
with open("README.md", 'r') as f:
    long_description = f.read()
setup(
    name=__package__,
    packages=[__package__],
    install_requires=[
        "numpy>=1.21.2",
        "transformers>=4.0.0",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    description='',
    author='ArgmaxML',
    author_email='splade@argmaxml.com',
    url='https://github.com/argmaxml/splade4elastic',
    keywords=[],
    classifiers=[],
    extras_require = {
        'faiss': ['faiss-cpu>=1.7.1'],
        'elasticsearch': ['elasticsearch>=8.5.0'],

    }
)

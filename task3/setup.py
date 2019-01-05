import setuptools

setuptools.setup(
    name="k-means",
    version="0.0.1",
    author="Petr Shumilov",
    author_email="petr1shum@gmail.com",
    description="Implementation of k-means",
    url="https://github.com/petr-shumilov/stochastic-course",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'sklearn',
        'Pillow',
    ],
)


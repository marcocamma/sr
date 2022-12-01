from setuptools import setup,find_packages

def get_version():
    with open("sr/__init__.py", "r") as fid:
        lines = fid.readlines()
    version = None
    for line in lines:
        if "version" in line:
            version = line.rstrip().split("=")[-1].lstrip()
    if version is None:
        raise RuntimeError("Could not find version from __init__.py")
    version = version.strip("'").strip('"')
    return version


setup(name='sr',
      version=get_version(),
      description='Simple utilities for synchrotron radiation',
      url='https://github.com/marcocamma/sr',
      author='marco cammarata',
      author_email='marcocammarata@gmail.com',
      license='MIT',
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'h5py',
          'datastorage',
          'xraydb',
          'sympy',
      ],
      zip_safe=False)

from setuptools import setup, find_packages
setup(name = "molpro", version = 0.1,
      author="Boltzmann Labs",
      author_email="contact@boltzmann.co",
      description="A python package for 3d structure based drug design and validation of molecules",
      url="https://github.com/boltzmannlabs/molpro",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License",
        "Operating System :: OS Independent",],
      packages = find_packages()
     )

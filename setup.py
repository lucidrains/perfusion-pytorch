from setuptools import setup, find_packages

setup(
  name = 'perfusion-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.16',
  license='MIT',
  description = 'Perfusion - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/perfusion-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'memory editing',
    'text-to-image'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'open-clip-torch>=2.0.0,<3.0.0',
    'opt-einsum',
    'torch>=2.0'
  ],
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

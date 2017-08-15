from setuptools import setup

setup(name='magpy',
      version='0.1',
      description='Model Analysis with Gaussian Processes in Python',
      url='http://github.com/samcoveney/maGPy',
      author='Sam Coveney',
      author_email='coveney.sam@gmail.com',
      license='GPL-3.0+',
      packages=['magpy'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'future',
      ],
      zip_safe=False)

from setuptools import setup,find_packages

setup(name='github_analysis',
      version='0.1',
      description='This project aims to understand how people are currently using GitHub, with the eventual goal of providing recommendation on an easy-to-use alternative to Git.',
      url='https://github.com/UBC-MDS/RStudio-GitHub-Analysis',
      author='Juno Chen, Ian Flores, Rayce Rossum, Richie Zitomer',
      license='BSD 2-Clause',
      packages=find_packages('github_analysis'),
      zip_safe=False)

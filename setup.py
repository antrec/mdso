from setuptools import setup, find_packages

setup(name='mdso',
      version='0.1',
      description='Multi dimensional spectral ordering',
      long_description='implements algorithm from arXiv...',
      classifiers=[
        'Development Status :: 3 - Alpha',
        # 'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        # 'Topic :: Text Processing :: Linguistic',
      ],
      # keywords='funniest joke comedy flying circus',
      # url='http://github.com/storborg/funniest',
      # author='Flying Circus',
      # author_email='flyingcircus@example.com',
      # license='MIT',
      # packages=['mdso', 'mdso.data', 'mdso.evaluate'],
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib'
      ],
      include_package_data=True,
      zip_safe=False)

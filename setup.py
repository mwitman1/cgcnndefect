from setuptools import setup

setup(name='cgcnndefect',
      version='0.1',
      description='Custom cgcnn version with various updates',
      author='Matthew Witman',
      author_email='mwitman1@sandia.gov',
      license='MIT',
      packages=['cgcnndefect'],
      zip_safe=False,
      python_requires='>3.6',
      install_requires=[
        'pymatgen',
        'ase',
        'torch',
        'numpy',
        'scikit-learn',
      ],
      entry_points={
          'console_scripts': ['cgcnn-defect-train=cgcnndefect.command_line_train:main',
                              'cgcnn-defect-predict=cgcnndefect.command_line_predict:main'],
      },
    )


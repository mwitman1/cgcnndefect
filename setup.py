from setuptools import setup

setup(name='cgcnniap',
      version='0.1',
      description='Custom cgcnn version with various updates',
      #url='http://github.com/mwitman1/heahydrides',
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
        'sklearn',
      ],
      entry_points={
          'console_scripts': ['cgcnn-defect-train=cgcnniap.command_line_train:main',
                              'cgcnn-defect-predict=cgcnniap.command_line_predict:main'],
      },
    )


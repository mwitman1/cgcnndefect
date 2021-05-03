from setuptools import setup

setup(name='cgcnniap',
      version='0.1',
      description='Custom cgcnn version with various updates',
      #url='http://github.com/mwitman1/heahydrides',
      author='Matthew Witman',
      author_email='mwitman1@sandia.gov',
      license='MIT',
      packages=['cgcnniap'],
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
          'console_scripts': ['cgcnn-iap-train=cgcnniap.command_line_train:main',
                              'cgcnn-iap-predict=cgcnniap.command_line_predict:main'],
      },
      #entry_points={
      #    'console_scripts': ['cgcnn-iap-train=cgcnn-iap.command_line_train:main'],
      #},
    )


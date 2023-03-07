from setuptools import setup

setup(name='resnet_dmp',
      description='Learn to unveil the stem of an occluded (from leaves) grapebunch',
      version='0.1',
      py_modules=['my_pkg'],
      author='Antonis Sidiropoulos',
      author_email='antosidi@ece.auth.gr',
      install_requires=['numpy==1.19',
                        'pybullet==3.2.0',
                        'opencv-python==4.5.1.48',
                        'matplotlib==3.3.3',
                        'scipy==1.4.1',
                        'PyYAML',
                        'scikit-learn',
                        'pyrealsense2',
                        'urdf-parser-py',
                        'tqdm',
      ]
)
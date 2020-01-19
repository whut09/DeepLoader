from setuptools import setup, find_packages  

setup(name='DeepLoader',
      version='0.4.8',
      description='Data loader for deep learning',
      url='https://github.com/cnzeki/deeploader',
      author='ZeKang Tian',
      author_email='zekang.tian@gmail.com',
      license='Apache License Version 2.0, January 2004',
      #packages=['deeploader, deeploader.dataset', 'deeploader.eval', 'deeploader.plats', 'deeploader.util'],
      packages = find_packages(),
      install_requires=[
            "numpy",
            "opencv-python",
            "future",
            "scipy",
            "scikit-image",
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',  # Again, pick a license
          'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
      ],
      keywords=['deep learning', 'image segmentation', 'image classification', 'batch prefetching','data loader', 'data augmentation']
      )

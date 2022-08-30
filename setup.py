import os
import setuptools
import sys


# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
sys.path.append(os.path.dirname(__file__))
import versioneer


setuptools.setup(
    name='openpifpaf_extras',
    version=versioneer.get_version(),
    license='',
    description='Extensions to OpenPifPaf',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/openpifpaf/openpifpaf_extras',

    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,

    python_requires='>=3.7',
    install_requires=[
        'timm>=0.4.9,<0.5',  # For Swin Transformer and XCiT
        'einops>=0.3',  # required for BotNet
        'openpifpaf>=0.13.4',
    ],
    extras_require={
        'dev': [
            'wheel',
        ],
        'test': [
            'pylint<2.9.4',  # avoid 2.9.4 and up for time.perf_counter deprecation warnings
            'pycodestyle',
            'pytest',
        ],
    },
)

from setuptools import setup, find_packages
from fraud.assets.config.config import cfg_item

setup(
    name=cfg_item("app", "name"),
    version=cfg_item("app", "version"),
    packages=find_packages(),
    install_requires=cfg_item("app", "install_requieres"),
    package_data={
        'fraud': [
            'assets/config/*.json',
            'assets/config/*.py',
            'assets/controller/*.py',
            'assets/examples/*.csv',
            'assets/images/*.ico',
            'assets/model/*.py',
            'assets/packages/predictors/*.py',
            'assets/packages/predictors/files/*.pkl',
            'assets/packages/transformers/*.py',
            'assets/packages/transformers/files/*.joblib',
            'assets/view/*.py'
        ],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'fraud-detector = fraud.__main__:main',
        ],
    },
    author=cfg_item("app", "author"),
    author_email=cfg_item("app", "author_email"),
    description='Description of your project',
    long_description=open('README.md').read(),
    url=cfg_item("app", "url"),
    classifiers=cfg_item("app", "classifiers")
)

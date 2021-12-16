from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f.readlines() if line.strip()]
    return requirements


def read_readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


def get_data_files():
    data_dir = Path('./vision_models/data')
    data_paths = [str(path).replace('\\', '/').replace('vision_models/', '')
                  for path in data_dir.glob('**/*') if path.is_file()]
    return data_paths


setup(
    name='vision_models',
    version='1.0.0',
    author='bbbnodx',
    author_email='ai-dev@ksk.co.jp',
    url='https://github.com/ksk-ai/vision-models',
    description='A PyTorch implementation of an OCR ensemble model.',
    long_description=read_readme(),
    license=license,
    install_requires=read_requirements(),
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "vision_models": get_data_files()
    }
)

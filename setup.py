from setuptools import setup,find_packages


with open("Requirements.txt") as f:
    requirements=f.read().splitlines()

if "-e ." in requirements:
    requirements.remove("-e .")


setup(
    name='ART_2_U',
    version='0.0.1',
    author='Astrokazgeot',
    author_email='sidskaz297@gmail.com',
    packages=find_packages(),
    install_requires=requirements
)
from setuptools import setup, find_packages

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


requirements = read_requirements("requirements.txt")

setup(
    name='isaac_phyrl_go2',
    version='1.0.0',
    author='Yihao Cai',
    license="MIT license",
    packages=find_packages(),
    author_email='yihao.cai@wayne.edu',
    description='Template Learning Machine for Safety-critical Environments',
    keywords=['unitree go2', 'mpc control', 'runtime learning'],
    install_requires=requirements
)

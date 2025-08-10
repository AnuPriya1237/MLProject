from setuptools import find_packages, setup


def get_requirements(file_path:str)->List[str]:
    "This function returns list of requirements"
    requirements=[]
    with open(file_path) as file_pth:
        requirements=file_pth.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements


setup(
    name='ML_PROJECT',
    version='0.0.1',
    author='APriya',
    author_email='priya7856anu@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
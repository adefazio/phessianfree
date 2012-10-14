from setuptools import * 

setup(
	name = 'phessianfree',
	version = '0.1',
	packages = find_packages(),
    install_requires=[
		'setuptools',
		'scipy>=0.10.0',
		'Theano>=0.5.0'
	],
	author = "Aaron Defazio",
	author_email = "aaron.defazio@anu.edu.au",
	licence = "BSD",
	keywords = "newton lbfgs optimization hessian hessianfree",
	url = "",
)

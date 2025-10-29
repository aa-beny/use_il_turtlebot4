from setuptools import find_packages, setup

package_name = 'navlearn_data_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='open',
    maintainer_email='open@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'collector = navlearn_data_tools.collector:main',
        'timecollector = navlearn_data_tools.timecollector:main',
        'dagger_collector = navlearn_data_tools.dagger_collector:main',
        
        ],
    },
)

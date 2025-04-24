
from setuptools import setup, find_packages

setup(
    name="threat-hunting-rag",
    version="0.1.0",
    packages=find_packages(exclude=['static*', 'templates*', 'attached_assets*']),
    python_requires=">=3.8",
    install_requires=[
        "flask",
        "werkzeug",
        "gunicorn"
    ]
)

from setuptools import setup, find_packages

setup(
    name='telco_churn_project',
    version='0.1.0',
    description='Churn analysis project - EDA and Modeling',
    author='Nicholas Tadeu Santos da Silva',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'xgboost',
        'shap',
        'joblib',
    ],
    python_requires='>=3.8',
)
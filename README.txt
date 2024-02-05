This reposiroty includes the features and code required to reproduce the MOOC fairness analysis published in XXX.

Specifically, the purpose is to study the biases of a et of machine learning (ML) classifiers against various demographics: Age, Gender, First language, Country of residence (Africa vs Europe), JobStatus, Having Children, Levls of education, Parental education, First time in a MOOC, Enrolled in cohort by a university.

In the feature folder, the input CSV files include these demographics values, along with the target labels to be predicted by the ML models (grade/passing the MOOC) and the behavioral features (number of actions performed in the edX MOOC platform).

The ML models are predefined in src/ML_models.py are are based on scicit-learn. In ML_models.py the hyperparameter default values and range of values of gris search are explicitely defined.

The main file is train_models.py, where the ML classifiers are initialized and trained. The code to compute the fairness metrics is in Module/

Requirements:
- Check environment.yml

Run:
- python3 train_models.py


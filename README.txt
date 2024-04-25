This repository includes the data and code required to reproduce the MOOC fairness analysis published in the Proceedings of the 25th International Conference on Artificial Intelligence in Education (AIED'2024).

Specifically, the purpose is to study the biases of machine learning (ML) classifiers meant to predict MOOC completion against various demographics: Age, Gender, First language, Country of residence (Africa+Haiti vs Western Europe), JobStatus, Having Children, Levels of education, Levels of Parental education, First time attending a MOOC, Enrolled in cohort by a university.

In the feature folder, the input CSV files include these demographics values, along with the target labels to be predicted by the ML models (grade/passing the MOOC) and the behavioral features (number of actions performed in the edX MOOC platform) for training pruposes.

The ML models are predefined in src/ML_models.py and are based on scicit-learn. In ML_models.py the hyperparameter default values and range of values of grid search are explicitly defined.

The main file is train_models.py, where the ML classifiers are initialized and trained. The code to compute the fairness metrics is in Module/

Requirements:
- Check environment.yml

Run:
- python3 train_models.py

If you reuse parts of the code and/or data, please cite the following paper: 
- Sébastien Lallé, François Bouchet, Mélina Verger and Vanda Luengo. 2024. Fairness of MOOC Completion Predictions Across Demographics and Contextual Variables. Proceedings of the 25th International Conference on Artificial Intelligence in Education, Recife, Brazil: Springer.

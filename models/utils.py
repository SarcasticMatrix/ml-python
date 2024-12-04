from sklearn.model_selection import cross_val_score
import numpy as np

def cv_zoom(
        model, 
        X_train: np.array, 
        y_train: np.array, 
        num_folds: int
    ):
    """
    Print un tableau avec les métriques du model sur chaque fold.
    """

    recall_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='recall')
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='accuracy')
    precision_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='precision')

    # Affiche les scores (recall, accuracy, precision) pour chaque fold
    print(f"Fold  : {'precision':10} | {'accuracy':10} | {'recall':10}")
    for i, (recall, accuracy, pecision) in enumerate(zip(recall_scores, accuracy_scores, precision_scores)):
        recall = round(recall,3)
        accuracy = round(accuracy,3) 
        precision = round(pecision,3)

        print(f"Fold {i+1}: {precision:10} | {accuracy:10} | {recall:10}")


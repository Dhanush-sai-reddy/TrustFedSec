import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from src.model import predict_proba

def evaluate(X, y, theta):
    prob = predict_proba(X, theta); pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'loss': float(log_loss(y, prob, labels=[0,1])), 'accuracy': float(accuracy_score(y,pred)), 'precision': float(precision_score(y,pred,zero_division=0)), 'recall': float(recall_score(y,pred,zero_division=0)), 'macro_f1': float(f1_score(y,pred,average='macro',zero_division=0)), 'false_positive_rate': float(fp/(fp+tn)) if fp+tn else 0.0, 'attack_recall': float(tp/(tp+fn)) if tp+fn else 0.0}

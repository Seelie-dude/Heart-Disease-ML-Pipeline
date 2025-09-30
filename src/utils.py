from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
class Plotter:
    def __init__(self, data):
        self.data=data

    def histogram(self, column):
        plt.hist(self.data[column], color= 'm', edgecolor= 'black')
       
    def boxplot(self, column_x, column_y):
        plt.boxplot([self.data[self.data[column_x]==val][column_y] for val in self.data[column_x].unique()])
      
    def heatmap(self):
        sns.heatmap(self.data.corr(), annot = True, cmap = 'coolwarm' )
        
    def scatter(self, column_x, column_y, target):
        if isinstance(target, pd.DataFrame):
            target = target.values.ravel()  # flatten DataFrame to 1D array
        elif isinstance(target, pd.Series):
            target = target.values
        plt.scatter(self.data[:, column_x], self.data[:, column_y], c=target, cmap='viridis')
        plt.colorbar(label="Target")

    def label(self, xlabel, ylabel, title):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

class N_selector:
    def __init__(self, data, threshold: float):
        self.data = data
        self.threshold = threshold
        self.cumulative_variance = None
        self.pca = PCA()
        self.pca.fit(self.data)
    def calc_n(self):
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(self.cumulative_variance >= self.threshold) + 1
        return n_components

class Preprocessor:
    def __init__(self):
        # Define column groups once
        self.categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        self.numerical_cols   = ['age', 'trestbps', 'chol', 'thalach', 'ca']

        # Initialize encoders/scalers
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.scaler  = MinMaxScaler()

    def fillna(self, data):
        data = data.copy()
        data.fillna(data.mode().iloc[0], inplace=True)
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        "Fit on data and return processed DataFrame"
        # Encode categorical
        encoded = self.encoder.fit_transform(data[self.categorical_cols])
        cats_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.categorical_cols))

        # Scale numerical
        scaled = self.scaler.fit_transform(data[self.numerical_cols])
        nums_df = pd.DataFrame(scaled, columns=self.numerical_cols)

        # Final processed DataFrame
        self.data_final = pd.concat([nums_df, cats_df], axis=1)
        return self.data_final
        
    def transform (self, data: pd.DataFrame) -> pd.DataFrame:
        # Encode categorical
        encoded = self.encoder.transform(data[self.categorical_cols])
        cats_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.categorical_cols))

        # Scale numerical
        scaled = self.scaler.transform(data[self.numerical_cols])
        nums_df = pd.DataFrame(scaled, columns=self.numerical_cols)

        # Final processed DataFrame
        self.data_final = pd.concat([nums_df, cats_df], axis=1)
        return self.data_final



class Train:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.log_clf = None
        self.dt_clf = None
        self.rf_clf = None
        self.svm_clf = None
    
    def logistic_reg(self):
        self.log_clf = LogisticRegression(max_iter=1000, C=1.0)
        self.log_clf.fit(self.X_train, self.y_train)
        return self   # return Train object for chaining

    def decision_tree(self):
        self.dt_clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
        self.dt_clf.fit(self.X_train, self.y_train)
        return self

    def random_forest(self):
        self.rf_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        self.rf_clf.fit(self.X_train, self.y_train)
        return self

    def svm(self):
        self.svm_clf = svm.SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
        self.svm_clf.fit(self.X_train, self.y_train)
        return self

    def get_metrics(self, model):
        """Pass in one of the trained models (e.g. self.svm_clf)"""
        y_pred = model.predict(self.X_test)
        
        metrics = pd.Series({
        "accuracy": model.score(self.X_test, self.y_test),
        "precision": precision_score(self.y_test, y_pred, average='weighted'),
        "recall": recall_score(self.y_test, y_pred, average='weighted'),
        "f1 score": f1_score(self.y_test, y_pred, average='weighted')
        })
        
        return metrics

    def plot_roc(self, model, model_name= 'Model'):
        y_score = model.predict_proba(self.X_test)[:, 1] 
        fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
        roc_auc =auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC for {model_name}")
        plt.legend()
        plt.show()
class Get_K:
    def __init__(self,X_test, k_range:int):
        self.X_test = X_test
        self.WCSS = []
        self.K = range(1, k_range)

    def elbow(self):
        for k in self.K:
            kmeanModel = KMeans(n_clusters=k, random_state=42).fit(self.X_test)
            self.WCSS.append(kmeanModel.inertia_)

        plt.plot(self.K, self.WCSS, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

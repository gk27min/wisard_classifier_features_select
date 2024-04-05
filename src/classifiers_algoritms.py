import wisardpkg as wsd
from sklearn.svm import SVC as svc
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class WisardClassifier:
    def __init__(self, ram=62, min_score=0.5, threshold=1000, discriminator_limit=5):
        self.model = wsd.ClusWisard(ram, min_score, threshold, discriminator_limit)
        
    def predict(self, X):
        ds_X = wsd.DataSet(X)
        predicted_labels = self.model.classify(ds_X)
        return predicted_labels

    def fit(self, X, y, **params):
        ds_train = wsd.DataSet(X, y)
        self.model.train(ds_train)
    
    def score(self, X, y):
        ds_test = wsd.DataSet(X, y)
        y_pred = self.model.classify(ds_test)
        return f1_score(y, y_pred, average='macro')
    
    def get_discriminators(self):
        discriminators = self.model.getMentalImages()
        return discriminators

class SVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svc(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def fit(self, X, y, **params):
        self.model.fit(X, y, **params)
        
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred,  average='macro')

    def get_params(self, deep=True):
        return {'kernel': self.kernel, 'C': self.C, 'gamma': self.gamma}
    
class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        
    def predict(self, X):
        predicted_labels = self.model.predict(X)
        return predicted_labels

    def fit(self, X, y, **params):
        self.model.fit(X, y, **params)
    
    def score(self, X, y):
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred, average='macro')

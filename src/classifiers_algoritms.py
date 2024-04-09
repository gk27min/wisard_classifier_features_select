import wisardpkg as wsd
from sklearn.svm import SVC as svc
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class WisardClassifier(BaseEstimator):
    def __init__(self, ram=62, min_score=0.5, threshold=1000, discriminator_limit=5):
        self.ram = ram
        self.min_score = min_score
        self.threshold = threshold
        self.discriminator_limit = discriminator_limit
        self.model = wsd.ClusWisard(ram, min_score, threshold, discriminator_limit)
        
    def fit(self, X, y, **params):
        ds_train = wsd.DataSet(X, y)
        self.model.train(ds_train)
    
    def predict(self, X):
        ds_X = wsd.DataSet(X)
        predicted_labels = self.model.classify(ds_X)
        return predicted_labels

    def score(self, X, y):
        ds_test = wsd.DataSet(X, y)
        y_pred = self.model.classify(ds_test)
        return f1_score(y, y_pred, average='macro')
    
    def get_discriminators(self):
        discriminators = self.model.getMentalImages()
        return discriminators

    def get_params(self, deep=True):
        return {
            'ram': self.ram,
            'min_score': self.min_score,
            'threshold': self.threshold,
            'discriminator_limit': self.discriminator_limit
        }

    def set_params(self, **params):
        self.ram = params.get('ram', self.ram)
        self.min_score = params.get('min_score', self.min_score)
        self.threshold = params.get('threshold', self.threshold)
        self.discriminator_limit = params.get('discriminator_limit', self.discriminator_limit)
        
        self.model = wsd.ClusWisard(self.ram, self.min_score, self.threshold, self.discriminator_limit)
        
        return self

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

    def get_params(self, deep=True):
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)

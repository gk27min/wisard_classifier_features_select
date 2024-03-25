import wisardpkg as wsd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

class WisardClassifier:
    def __init__(self, ram=62, min_score=0.5, threshold=1000, discriminator_limit=5):
        self.model = wds.ClusWisard(ram, min_score, threshold, discriminator_limit)
        
    def set_train(self, train, labels):
        self.ds_train = wds.DataSet(train, labels)
        
    def set_test(self, test, labels):
        self.ds_test = wds.DataSet(test, labels)
        
    def train(self):
        self.model.train(self.ds_train)
        
    def predict(self):
        predicted_labels = self.model.classify(self.ds_test)
        return predicted_labels

    def fit(self, X, y):
        self.ds_train = wds.DataSet(X, y)
        self.model.train(self.ds_train)
    
    def score(self, X, y):
        self.ds_test = wds.DataSet(X, y)
        y_pred = self.model.classify(self.ds_test)
        return f1_score(y, y_pred)
    
    def get_discriminators(self):
        discriminators = self.model.getMentalImages()
        return discriminators

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        
    def set_train(self, train, labels):
        self.train_data = train
        self.train_labels = labels
        
    def set_test(self, test, labels):
        self.test_data = test
        self.test_labels = labels
        
    def train(self):
        self.model.fit(self.train_data, self.train_labels)
        
    def predict(self):
        predicted_labels = self.model.predict(self.test_data)
        return predicted_labels

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def score(self, X, y):
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred)

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        
    def set_train(self, train, labels):
        self.train_data = train
        self.train_labels = labels
        
    def set_test(self, test, labels):
        self.test_data = test
        self.test_labels = labels
        
    def train(self):
        self.model.fit(self.train_data, self.train_labels)
        
    def predict(self):
        predicted_labels = self.model.predict(self.test_data)
        return predicted_labels

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def score(self, X, y):
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred)

from collections import Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes):
        self._set_priors(train_classes)
        discrete_features = self.data_discretization(train_features)
        self._set_likelihoods(discrete_features, train_classes)

    def _set_priors(self, train_classes):
        for single_class in np.unique(train_classes):
            class_prob = np.sum(train_classes == single_class) / len(train_classes)
            self.priors[single_class] = class_prob

    def _set_likelihoods(self, discrete_train_features, train_classes):
        for single_class in np.unique(train_classes):
            self.likelihoods[single_class] = {}
            features = discrete_train_features[train_classes == single_class]

            num_of_features = discrete_train_features.shape[1]
            for feature_index in range(num_of_features):
                self.likelihoods[single_class][feature_index] = {}
                feature = features[:, feature_index]

                for bin_num in range(1, 5):
                    feature_for_single_bin = feature[feature == bin_num]
                    feature_prob = len(feature_for_single_bin) / len(feature)

                    self.likelihoods[single_class][feature_index][bin_num] = feature_prob

    @staticmethod
    def data_discretization(data: np.ndarray) -> np.ndarray:
        copied_data = data.copy()
        num_of_features = data.shape[1]
        for feature_index in range(num_of_features):
            feature_values = data[:, feature_index]
            bins_ranges = np.linspace(np.min(feature_values), np.max(feature_values), 4)
            feature_discrete = np.digitize(feature_values, bins_ranges)
            copied_data[:, feature_index] = feature_discrete
        return copied_data

    def predict(self, samples: np.ndarray) -> np.ndarray:
        discrete_samples = self.data_discretization(samples)

        predictions = []
        for sample in discrete_samples:
            posterior_probabilities = {}
            for single_class in self.priors:
                posterior_probability = self.priors[single_class]

                for feature_index, bin_num in enumerate(sample):
                    posterior_probability *= self.likelihoods[single_class][feature_index][bin_num]

                posterior_probabilities[single_class] = posterior_probability
            predicted_class = max(posterior_probabilities, key=lambda k: posterior_probabilities[k])
            predictions.append(predicted_class)

        return np.array(predictions)


class GaussianNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes):
        pass

    @staticmethod
    def normal_dist(x, mean, std):
        pass

    def predict(self, sample):
        pass


if __name__ == "__main__":
    bayes_classifier = NaiveBayes()
    bayes_classifier.build_classifier(x_train, y_train)
    print("Naive Bayes Classifier")
    print("Predicted classes:\t", end="")
    print(bayes_classifier.predict(x_test))
    print("Aktual classes:   \t", end="")
    print(y_test)

    print("Gaussian Naive Bayes Classifier")
    print("TBA")

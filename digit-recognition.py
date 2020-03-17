from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import misc

digits = datasets.load_digits() # This is dictionary

'''
plt.gray()
plt.matshow(digits.images[100]) 
print(digits.target[100])
plt.show()
'''

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))
# The new shape should be compatible with the original shape.
'''One shape dimension can be -1. In this case, 
the value is inferred from the length of the array and remaining dimensions.'''

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = plt.imread("C:\\Users\\george\\Desktop\\1.png")
#image = rgb2gray(image)
length = len(image)
image1 = np.reshape(image,(1,64))
#image1 = image1[:51]
result = classifier.predict(image1)




print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()

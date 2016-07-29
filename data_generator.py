if __name__ == '__main__':
    from sklearn.datasets import make_moons, make_classification
    from sklearn.cross_validation import train_test_split
    l = 200
    X, y = make_moons(n_samples=l, noise=0.3, random_state=0)
    y = y*2 - 1
    #X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1) # linearly separable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    from sklearn.datasets import dump_svmlight_file
    with open('mydata', 'w') as f:
        dump_svmlight_file(X_train, y_train, f, zero_based=False)
    with open('mydata.t', 'w') as f:
        dump_svmlight_file(X_test, y_test, f, zero_based=False)

    from sklearn.svm import SVC
    clf = SVC(gamma='auto', C=1)
    print('LibSVM, rbf kernel')
    clf.fit(X_train, y_train)
    print 'accuracy:', clf.score(X_test, y_test)
    print

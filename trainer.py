import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import post_processing as pp
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=570)


def return_nonstring_col(data_cols):
	cols_to_keep=[]
	train_cols = []
	for col in data_cols:
		if col != 'URL' and col != 'host' and col != 'path':
			cols_to_keep.append(col)
			if col != 'malicious' and col != 'result':
				train_cols.append(col)
	return [cols_to_keep, train_cols]


def svm_classifier(train, query, train_cols):
	
	clf = svm.SVC(gamma='auto')
	# print(train[train_cols])
	train[train_cols] = preprocessing.scale(train[train_cols])
	query[train_cols] = preprocessing.scale(query[train_cols])
	x = train[train_cols]
	y = train['malicious']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_train, y_train, cv=5)
	print('Estimated score SVM: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

	predictions = clf.predict(x_test)
	y_test = list(y_test)
	accuracy = pp.find_accuracy(y_test, predictions)
	print("Accuracy:", accuracy)
	confusion_matrix = pp.find_confusion_matrix(predictions, y_test)

	pp.plotConfusionMatrix(y_test, predictions, confusion_matrix,
						   normalize=True,
						   title=None,
						   cmap=None, plot=True)


def forest_classifier(train, query, train_cols):
	x = train[train_cols]
	y = train['malicious']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	rf = RandomForestClassifier(n_estimators=150)

	print(rf.fit(x_train, y_train))
	scores = cross_val_score(rf, x_train, y_train, cv=5)
	print('Estimated score RandomForestClassifier: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

	predictions = rf.predict(x_test)
	y_test = list(y_test)
	accuracy = pp.find_accuracy(y_test, predictions)
	print("Accuracy:", accuracy)
	confusion_matrix = pp.find_confusion_matrix(predictions, y_test)

	pp.plotConfusionMatrix(y_test, predictions, confusion_matrix,
						   normalize=True,
						   title=None,
						   cmap=None, plot=True)


def KNN_classifier(train, query, train_cols):
	x = train[train_cols]
	y = train['malicious']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	model = KNeighborsClassifier(n_neighbors=3)
	model.fit(x_train, y_train)
	# query['result'] = model.predict(x_test)
	predictions = model.predict(x_test)
	y_test = list(y_test)
	accuracy = pp.find_accuracy(y_test, predictions)
	print("Accuracy:", accuracy)
	confusion_matrix = pp.find_confusion_matrix(predictions, y_test)

	pp.plotConfusionMatrix(y_test, predictions, confusion_matrix,
					   		normalize=True,
					   		title=None,
					   		cmap=None, plot=True)


def NN_classifier(train, query, train_cols):
	x = train[train_cols]
	y = train['malicious']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	# x_train = x
	# y_train = y

	sc_x = StandardScaler()
	x_train = sc_x.fit_transform(x_train)
	x_test = sc_x.transform(x_test)

	# initialising the ANN
	classifier = Sequential()

	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=len(train_cols)))

	# Adding the second hidden layer
	classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

	# Adding the third hidden layer
	classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

	# Adding the fourth hidden layer
	classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

	# Adding the fifth hidden layer
	classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

	# Adding the output layer
	classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

	# compiling the ANN
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# Fitting the Ann to the Training set
	history = classifier.fit(x_train, y_train, batch_size=20, epochs=200, validation_data=(x_test, y_test))
	pp.plot_learning_curves(history)

	# predicting the test set results
	y_pred = classifier.predict(x_test)
	y_pred = (y_pred > 0.5)
	y_pred = (y_pred.astype(int)).tolist()
	y_test = list(y_test)
	y_pred = sum(y_pred, [])
	accuracy = pp.find_accuracy(y_test, y_pred)
	print("Accuracy:", accuracy)
	confusion_matrix = pp.find_confusion_matrix(y_pred, y_test)

	pp.plotConfusionMatrix(y_test, y_pred, confusion_matrix,
						   normalize=True,
						   title=None,
						   cmap=None, plot=True)
	# y_predict = y_pred.astype(int)
	# query['result'] = y_predict
	# making confusion matrix
	# from sklearn.metrics import confusion_matrix
	# cm = confusion_matrix(y_test, y_pred)
	# print('Confusion Matrix:', cm)


def train(db, test_db, classifier):
	
	query_csv = pandas.read_csv(test_db)
	cols_to_keep, train_cols = return_nonstring_col(query_csv.columns)
	# query=query_csv[cols_to_keep]

	train_csv = pandas.read_csv(db)
	cols_to_keep, train_cols = return_nonstring_col(train_csv.columns)
	train = train_csv[cols_to_keep]
	if classifier == 'NN':
		NN_classifier(train_csv, query_csv, train_cols)
	elif classifier == 'knn':
		KNN_classifier(train_csv, query_csv, train_cols)
	elif classifier == 'svm':
		svm_classifier(train_csv, query_csv, train_cols)
	elif classifier == 'rf':
		forest_classifier(train_csv, query_csv, train_cols)
	# accuracy = pp.find_accuracy(query_csv['malicious'], query_csv['result'])
	# print("Accuracy:", accuracy)
	# confusion_matrix = pp.find_confusion_matrix(query_csv['result'], query_csv['malicious'])
	# pp.plotConfusionMatrix(query_csv['malicious'], query_csv['result'], confusion_matrix,
	# 				   		normalize=True,
	# 				   		title=None,
	# 				   		cmap=None, plot=True)


	# svm_classifier(train_csv, query_csv, train_cols)
	# accuracy = pp.find_accuracy(query_csv['malicious'], query_csv['result'])
	# confusion_matrix = pp.find_confusion_matrix(query_csv['result'], query_csv['malicious'])
	# pp.plotConfusionMatrix(query_csv['malicious'], query_csv['result'], confusion_matrix,
	#					   normalize=True,
	#					   title=None,
	#					   cmap=None, plot=True)
	#
	# forest_classifier(train_csv, query_csv, train_cols)
	#
	# accuracy = pp.find_accuracy(query_csv['malicious'], query_csv['result'])
	# confusion_matrix = pp.find_confusion_matrix(query_csv['result'], query_csv['malicious'])
	# pp.plotConfusionMatrix(query_csv['malicious'], query_csv['result'], confusion_matrix,
	# 				normalize=True,
	# 				title=None,
	# 				cmap=None, plot=True)






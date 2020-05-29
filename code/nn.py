import sklearn



import numpy as np



from sklearn.neural_network import MLPRegressor



from sklearn.ensemble import RandomForestRegressor







def main():







	for i in range(1, 8) : 







		problem = i



		if problem ==	1:



			traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")



			testdata	= np.loadtxt("Data_OneStepAhead/Lazer/test.txt")	#



			name	= "Lazer"



		if problem ==	2:



			traindata = np.loadtxt(  "Data_OneStepAhead/Sunspot/train.txt")



			testdata	= np.loadtxt( "Data_OneStepAhead/Sunspot/test.txt")	#



			name	= "Sunspot"



		if problem ==	3:



			traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")



			testdata	= np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #



			name	= "Mackey"



		if problem ==	4:



			traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")



			testdata	= np.loadtxt("Data_OneStepAhead/Lorenz/test.txt")  #



			name	= "Lorenz"



		if problem ==	5:



			traindata = np.loadtxt( "Data_OneStepAhead/Rossler/train.txt")



			testdata	= np.loadtxt( "Data_OneStepAhead/Rossler/test.txt")	#



			name	= "Rossler"



		if problem ==	6:



			traindata = np.loadtxt("Data_OneStepAhead/Henon/train.txt")



			testdata	= np.loadtxt("Data_OneStepAhead/Henon/test.txt")	#



			name	= "Henon"



		if problem ==	7:



			traindata = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt") 



			testdata	= np.loadtxt("Data_OneStepAhead/ACFinance/test.txt")	#



			name	= "ACFinance"



	



		x_train = traindata[:,0:3]



		y_train = traindata[:,4]



		x_test = testdata[:,0:3]



		y_test = testdata[:,4]



		



		mlp_adam = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='adam', alpha=0.1,max_iter=100000, tol=0)



		mlp_adam.fit(x_train,y_train)



		train_acc = np.sqrt(((mlp_adam.predict(x_train) - y_train)**2).sum())



		test_acc = np.sqrt(((mlp_adam.predict(x_test) - y_test)**2).sum())



		print(name,train_acc, test_acc)



		



		mlp_sgd = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='sgd', alpha=0.1,max_iter=100000, tol=0)



		mlp_sgd.fit(x_train,y_train)



		train_acc = np.sqrt(((mlp_sgd.predict(x_train) - y_train)**2).sum())



		test_acc = np.sqrt(((mlp_sgd.predict(x_test) - y_test)**2).sum())



		print(name,train_acc, test_acc)



		



		rf = RandomForestRegressor()



		rf.fit(x_train,y_train)



		train_acc = np.sqrt(((rf.predict(x_train) - y_train)**2).sum())



		test_acc = np.sqrt(((rf.predict(x_test) - y_test)**2).sum())



		print(name,train_acc, test_acc)







if __name__ == "__main__": main()

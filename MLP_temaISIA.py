from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('kohkiloyeh.xlsx')
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values
print(dataset.describe().transpose())
labelencoder_X_0 = LabelEncoder() #coloana dregree
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0]) #0=high 1=low 2=medium
labelencoder_X_1 = LabelEncoder() #cloana caprice
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #0=left 1=middle 2=right
labelencoder_X_2 = LabelEncoder() #coloana topic 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #0=impressior 1=news 2=political 3=scientific 4=tourism
labelencoder_X_3 = LabelEncoder() #coloana local media turnover
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3]) #0=no 1=yes 
labelencoder_X_4 = LabelEncoder() #coloana political and social space
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4]) #0=no 1=yes
labelencoder_y = LabelEncoder() #coloana pro bloggers
y = labelencoder_y.fit_transform(y) #0=no 1=yes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
mlp = MLPClassifier(hidden_layer_sizes=(3),max_iter=100, learning_rate_init=0.01)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
k=0
for i  in range(0,len(y_pred)) :
    if y_pred[i] == y_test[i]:
        k+=1
accuracy = float(k)/float(len(y_pred))
s = "Acuratetea este de " + str(accuracy*100) +"%"
print(s)






import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score

#1) READ AND EXTRACT DATA

#1.1) get data out of the csv file
dataframe = pandas.read_csv("./RealMedicalData.csv",sep=';',decimal=b',')

listColNames = list(dataframe.columns)


#1.2) extract X and Y as numpy arrays

XY = dataframe.values
ColNb_Y = listColNames.index('Disease progression')


Y = XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only

n = np.shape(X)[0]
print(n)
thresh = n // 2
X_train = X_scaled[thresh:]
Y_train = Y[thresh:]
X_test = X_scaled[:thresh]
Y_test = Y[:thresh]

#2) EXPLORE THE DATA

for Col in range(len(listColNames)):
    plt.plot(X[:,Col],Y[:],'.')
    plt.xlabel(listColNames[Col])
    plt.ylabel('Disease progression')
    # plt.show()

#3) PERFORM THE REGRESSION

#3.1) ridge regression
alphas = np.linspace(0.01, 100, 1000)
r2_score_ridge_max = -9000
alpha = 0
alpha_optim = 0

from sklearn.linear_model import Ridge

for alpha in alphas:
    ridge_regressor = Ridge(alpha=alpha, fit_intercept=True)
    ridge_regressor.fit(X_train,Y_train)
    Y_pred_ridge = ridge_regressor.predict(X_test)
    r2_score_ridge = r2_score(Y_test, Y_pred_ridge)
    if r2_score_ridge > r2_score_ridge_max:
        r2_score_ridge_max = r2_score_ridge
        alpha_optim = alpha

print('alpha=', alpha_optim, 'r2=', r2_score_ridge_max)
# print('Beta values')
# for Col in range(len(listColNames)):
#   print('-> '+listColNames[Col]+': '+str(ridge_regressor.coef_[0,Col]))

#3.2) Lasso regression

# from sklearn.linear_model import Lasso
#
# lasso_regressor=Lasso(alpha=0.5, fit_intercept=True)
#
# lasso_regressor.fit(X_scaled,Y)
#
# print('Beta values')
# for Col in range(len(listColNames)):
#   print('-> '+listColNames[Col]+': '+str(lasso_regressor.coef_[Col]))





#QUESTION 1: Ouvrir le fichier RealMedicalData.csv (par exemple avec libreoffice) puis comprendre
#            chaque instruction du code

#QUESTION 2: Trouver une bonne valeur de alpha pour la regression Ridge et Lasso en utilisant la
#            separant les observations en un jeu d'apprentissage et un jeu de validation
#              -> Est-ce que les deux mÃ©thodes ont un bon pouvoir de prediction (utiliser metrics.r2_score) ?


#QUESTION 3: Afin de comprendre le lien entre l'evolution de la maladie et les variables etudiees,
#            on sÃ©lectionne tout au plus 3 variables avec le lasso.
#            -> Utiliser une procedure de type 4-folds pour selectionner typiquement 3 variables
#            -> Les variables selectionnees sont elles stables ?
#            -> Est-ce que le modele garde un bon pouvoir de prediction lorsque avec 3 variables selectionnees


#QUESTION 4: Eventuellement, tester un algorithme de selection de type forward avec un critere BIC.
#            Comparer les variables selectionnees avec celles de la question 3

#inspire de http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# generation de donnee synthetiques...
np.random.seed(31)

#... definition de n et p
n = 75
p =  200   #remarque : n<200 => necessite de selectionner des variables

#... simulation de X
X = np.random.randn(n, p) #remarque : on ne tient pas en compte les $beta_0$

#... generation du vecteur beta dans lequel seules 10 valeurs sont non-nulles
beta = 3 * np.random.randn(p)
inds = np.arange(p)
np.random.shuffle(inds)
beta[inds[10:]] = 0

#... simulation de y
y = np.dot(X, beta) + (0.01 * np.random.normal(size=n))
# REMARQUE IMPORTANTE : y ne dÃ©pend que des variables i pour lesquelles beta[i] est non-nul

# ... coupe en deux les donnees en donnees d'apprentissage et donnes test
thresh=n // 2
X_train = X[thresh:]
y_train = y[thresh:]
X_test = X[:thresh]
y_test = y[:thresh]

# regression lineaire avec regularisation Lasso ...

#... regression
from sklearn.linear_model import Lasso

alpha = 0.1
lasso_regressor = Lasso(alpha=alpha)
lasso_regressor.fit(X_train, y_train)

y_pred_lasso = lasso_regressor.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

#... representation du resultat

# fig = plt.figure()
# plt.plot(y_test, y_pred_lasso, 'r.')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')
# plt.title('True vs predicted value (r2='+str(r2_score_lasso)+')')
# plt.show()
#
#
#
# fig = plt.figure()
# plt.plot(lasso_regressor.coef_, 'r.')
# plt.plot(beta, 'b.')
# plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
# plt.title('Coefficients de beta')
# plt.show()


#QUESTION 1:
#Jouez l'exemple avec n=100, n=75, n=50, n=25. Qu'en deduisez vous sur l'impact du nombre d'observations

#QUESTION 2:
#On garde n=75. Utiliser la validation croisee de type K-folds pour trouver le meilleur coefficient alpha
#au sens de R2.
#AIDE : Pour comprendre l'utilisation de K-folds sous sklearn vous pouvez jouer l'exemple ci-dessou

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet

data = np.random.randn(12,3)
kf = KFold(n_splits=3)

alphas = np.linspace(0.0001, 0.1, 1000)
r2_score_lasso_max = 0
alpha = 0

for alpha in alphas:
    for train, test in kf.split(X):
      X_train = X[train,:]
      y_train = y[train]
      X_test = X[test,:]
      y_test = y[test]

      lasso_regressor = Lasso(alpha=alpha)
      lasso_regressor.fit(X_train, y_train)

      y_pred_lasso = lasso_regressor.predict(X_test)
      r2_score_lasso = r2_score(y_test, y_pred_lasso)

      if r2_score_lasso > r2_score_lasso_max:
          r2_score_lasso_max = r2_score_lasso
          alpha_optim = alpha

lasso_regressor = Lasso(alpha=alpha_optim)
lasso_regressor.fit(X_train, y_train)

y_pred_lasso = lasso_regressor.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print('alpha=', alpha_optim, 'r2=', r2_score_lasso_max)

fig = plt.figure()
plt.plot(y_test, y_pred_lasso, 'r.')
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('True vs predicted value (r2='+str(r2_score_lasso_max)+')')
plt.show()

plt.plot(lasso_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta')
plt.show()


#QUESTION 3:
#EVENTUELLEMENT : Comparez les resultats LASSO avec ceux d'ElasticNet
#Utilisez : from sklearn.linear_model import ElasticNet
#    puis : enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.7)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('data/total.csv')
cont = []

for i, row in df.iterrows():
    cont.append(row['CreateEvent'] - row['DeleteEvent'] + row['PushEvent'] + row['CommitCommentEvent'] \
    + row['ReleaseEvent'] + row['PublicEvent'] + row['PullRequestReviewCommentEvent'])

contp = np.percentile(cont, 98)
df = df[['WatchEvent', 'IssueCommentEvent', 'IssuesEvent', 'MemberEvent', 'PullRequestEvent', 'ForkEvent']]
df['contribution'] = cont
df = df[(df['contribution'] <= contp) & (df['contribution'] >= 0)]

X = df.values[:,:-1]
y = df.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV 

def tune():

    cv_params = {'n_estimators': [500, 1000, 1500, 2000, 2500], 'learning_rate': [0.001, 0.01, 0.1, 0.5, 1], 
                 'max_depth': [3, 4, 5, 6], 'min_child_weight': [1, 2, 3]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    return optimized_GBM

optimized_GBM = tune()
optimized_GBM.fit(X_train, y_train)

print('best_param：{0}'.format(optimized_GBM.best_params_))
print('best_sccore:{0}'.format(optimized_GBM.best_score_))


other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = xgb.XGBRegressor(**other_params)
model.fit(X_train, y_train)
ans = model.predict(X_test)

error = [abs(ans[i] - y_test[i]) for i in range(len(ans))]
print(np.mean(error))
print([error[i] / y_test[i] for i in range(len(ans))])

plt.figure(figsize=(12,6))
plt.barh(range(len(model.feature_importances_)), model.feature_importances_, tick_label = ['WatchEvent', 'IssueCommentEvent', 'IssuesEvent', 'MemberEvent', 'PullRequestEvent', 'ForkEvent'])
plt.show()

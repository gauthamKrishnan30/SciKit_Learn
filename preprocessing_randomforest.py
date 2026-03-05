from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
data=load_iris()

X=data.data
y=data.target


X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

random_forest=RandomForestClassifier()
random_forest.fit(X_train,y_train)
B=random_forest.score (X_test,y_test)
print(B)
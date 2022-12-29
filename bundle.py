import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from data_remaker import feature_engineering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = df.drop('категория', axis=1)
X = pd.get_dummies(data=X, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 100 and
                    X_train[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

model = RandomForestClassifier(n_estimators=128, max_features=3)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])


model_mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(10, 30, 10),
                          learning_rate='constant', solver='adam', max_iter=1000)
model_rfc = RandomForestClassifier(n_estimators=128, max_features=3)
model_ada = AdaBoostClassifier(n_estimators=59)
alg_models = [model_mlp, model_rfc, model_ada]

def pipe_count(model, X, y, cv_num):
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    cv_score = cross_val_score(my_pipeline, X, y, cv=cv_num, scoring='accuracy')
    return f"{model} cross validation score is: {cv_score}"

for model in alg_models:
    print(pipe_count(model, X, y, 5))

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\uchen\PycharmProjects\Medicines\Medicine_Details.csv")

# Create the review score column for my training
data['review_score'] = (
    (data['Excellent Review %'] * 5 + data['Average Review %'] * 3 + data['Poor Review %'] * 1) /
    (data['Excellent Review %'] + data['Average Review %'] + data['Poor Review %'] + 1e-10)

)

# Label Encoding of categorical variables
label_encoder = {}
for column in ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Manufacturer']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoder[column] = le


# Save the label encoder
joblib.dump(label_encoder, 'label_encoder_composition.joblib')
joblib.dump(label_encoder, 'label_encoder_medicine_name.joblib')
joblib.dump(label_encoder, 'label_encoder_uses.joblib')
joblib.dump(label_encoder, 'label_encoder_side_effects.joblib')
joblib.dump(label_encoder, 'label_encoder_image_url.joblib')
joblib.dump(label_encoder, 'label_encoder_manufacturer.joblib')


# Feature and target variable (independent and dependent variable)
# To define the feature variables
X = data[['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Manufacturer',
          'Excellent Review %', 'Average Review %', 'Poor Review %']]

# To define the target variable
y = data['review_score']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state= 40)



# Models
models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - Mean Squared Error: {mse}')
    print(f'{name} - R2 Score: {r2}')

# Hyperparameter for Random Forest
rf_params = {
    'n_estimators': [5, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Use GridSearchCV to search for the best hyperparameters
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=5,
                              scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
rf_best_model = rf_grid_search.best_estimator_

print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")

# Save the best Random Forest Model
joblib.dump(rf_best_model, 'random_forest_model.joblib')

# Hyperparameter tuning for Gradient Boosting
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

gbr = GradientBoostingRegressor()

# Use GridSearchCV to search for the best hyperparameters
gb_grid_search = GridSearchCV(gbr, gb_params, cv=5, scoring='neg_mean_squared_error')
gb_grid_search.fit(X_train, y_train)

# Get the best parameter and the best model
gb_best_model = gb_grid_search.best_estimator_

print(f"Best Gradient Boosting Parameters: {gb_grid_search.best_params_}")

# Save the best Gradient Boosting Model
joblib.dump(gb_best_model, 'gradient_boosting_model.joblib')

# Save my model
joblib.dump(models, f'{name.lower().replace( " ", "_")}_model.joblib')



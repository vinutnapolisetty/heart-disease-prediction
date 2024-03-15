import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# Get column names from metadata
column_names = getattr(heart_disease.variables, 'feature_names', None)

# If feature_names is not available, you may need to explore the variables attribute
# column_names = heart_disease.variables  # Uncomment this line and explore the variables attribute

# Convert data to pandas DataFrame
df = pd.DataFrame(data=X, columns=column_names)

# Add the target column to the DataFrame
df['target'] = y

# Display the DataFrame

df['target'] = df['target'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})



# Impute missing values in the data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Initialize KMeans
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit KMeans to the data
kmeans.fit(X_scaled)

# Predict cluster labels
cluster_labels = kmeans.predict(X_scaled)


accuracy = accuracy_score(y, cluster_labels)


filename = 'heart-disease-prediction-kmeans-model.pkl'
pickle.dump(kmeans, open(filename, 'wb'))
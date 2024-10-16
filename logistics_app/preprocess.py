import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path):
    # Load the data
    data = pd.read_csv(csv_path)

    # Feature columns and target
    features = ['shipping_time', 'distance', 'weather', 'traffic_conditions']
    target = 'delay'

    # Convert categorical features to numerical values
    data['weather'] = data['weather'].map({'clear': 0, 'rain': 1, 'snow': 2})
    data['traffic_conditions'] = data['traffic_conditions'].map({'low': 0, 'medium': 1, 'high': 2})

    X = data[features]
    y = data[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

from sklearn.ensemble import RandomForestClassifier
import joblib
from src.preprocess import load_and_preprocess_data

def train_model():
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/diabetes.csv')
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'diabetes_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("Model training complete!")
    except Exception as e:
        print(f"Error in training model: {e}")

if __name__ == "__main__":
    train_model()

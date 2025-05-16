import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_clean_data():
    data = pd.read_csv("D:\Git\Cancer_Prediction\Data\data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def create_models(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest Performing Model: {best_model_name} ({best_accuracy:.2f} Accuracy)")
    return best_model, scaler, best_model_name

def main():
    data = get_clean_data()
    best_model, scaler, best_model_name = create_models(data)
    with open("D:\Git\Cancer_Prediction\Model\model.pkl", "wb") as f:
          pickle.dump(best_model, f)
    with open("D:\Git\Cancer_Prediction\Model\scaler.pkl", "wb") as f:
          pickle.dump(scaler, f)
    print(f"{best_model_name} Model Saved for Deployment!")

if __name__ == '__main__':
    main()
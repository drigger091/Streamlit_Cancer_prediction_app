from sklearn.preprocessing import StandardScaler
from data_clean import clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report


data = clean_data()

def create_model(data):

    try:
        # Data preparation
        X = data.drop('diagnosis', axis=1)
        Y = data['diagnosis']

        # Scaling the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Training the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Testing the model
        y_pred = model.predict(X_test)

        # Printing results
        print("Accuracy of the model:", accuracy_score(y_test, y_pred))
        print("Classification report:\n", classification_report(y_test, y_pred))

        # Return model and scaler
        return model, scaler

    except (KeyError, ValueError) as e:
        print("Error:", e)
        return None, None  # Returning None for both model and scaler in case of error


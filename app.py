# Importing necessary libraries
import streamlit as st  # Streamlit is used for creating interactive web apps
import pickle  # Pickle is used for loading the saved model
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer for text processing

# Load the pre-trained SVM model from a pickle file
model = pickle.load(open('svm_model.pkl', 'rb'))

# Load the TF-IDF vectorizer from a pickle file
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Set the background image and text color for the Streamlit app
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.istockphoto.com/id/1330904062/photo/person-with-a-hoody-typing-at-a-computer-in-the-dark-suspicious-online-behavior.jpg?s=1024x1024&w=is&k=20&c=WAZsFlg119oWrRDA0PV-xs611k8wGmvBryQazScaq4A=");
    background-size: 100vw 100vh;  # Make the background cover the whole screen
    background-position: center;  # Center the image
    background-repeat: no-repeat;  # Prevent the image from repeating
    color: #FFFFFF;  # Set the text color to white
}
</style>
"""
# Render the background image using HTML and CSS
st.markdown(background_image, unsafe_allow_html=True)

# Define the main function for the Streamlit app
def main():
    # Set the title of the app
    st.title('Cyberbullying Prediction')

    # Create a text area where users can enter a tweet
    user_input = st.text_area('Enter a tweet:')

    # Add a button to trigger the prediction
    if st.button('Predict'):
        # Transform the user's input using the loaded TF-IDF vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        # Make a prediction using the loaded SVM model
        prediction = model.predict(user_input_tfidf)

        # Define a dictionary to map encoded sentiment values to sentiment names
        sentiment_mapping = {
            1: "Religion",
            2: "Age",
            3: "Ethnicity",
            4: "Gender",
            5: "Other Cyberbullying",
            6: "Not Cyberbullying"
        }

        # Get the sentiment name based on the predicted encoded value
        predicted_sentiment = sentiment_mapping[prediction[0]]

        # Display the predicted sentiment
        st.write(f'Predicted sentiment: {predicted_sentiment}')

# Run the main function when the script is executed
if __name__ == '__main__':
    main()

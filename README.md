# Proverb-Recommendation-Engine-Website (AI & LLM)
# African Proverb Recommendation Engine

## Introduction
The African Proverbs Recommender is a web application built using Streamlit that provides users with recommendations of African proverbs based on a provided theme or mood. The recommendation engine utilizes pre-trained language models from the Hugging Face `transformers` library to compute semantic similarity between user input and a dataset of African proverbs.

## Features
- **Theme-based Proverb Recommendations:** Users can input a theme or mood, and the application recommends African proverbs related to the input.
- **Interactive Interface:** The web application provides a user-friendly interface where users can easily input their theme/mood and see recommended proverbs.
- **Warning about Data Limitations:** To maintain transparency, the application warns users about potential inaccuracies in recommendations due to the limited size of the dataset.
- **Random Proverb Display:** Users can see additional randomly selected proverbs for exploration and enjoyment.

## Recommendation Engine
The recommendation engine behind the application employs the following techniques and tools:
- **DistilBERT Model:** The engine utilizes a DistilBERT model from the Hugging Face `transformers` library to compute semantic similarity between the user input and proverbs in the dataset.
- **Cosine Similarity:** Cosine similarity is used to measure the similarity between embeddings of user input and proverbs.
- **Streamlit:** The recommendation engine is integrated into a Streamlit web application, providing a simple and interactive user interface for users to interact with.

## Data Collection and Processing
I gathered African proverbs and their meanings from various websites using web scraping techniques implemented with BeautifulSoup. After collecting the data, I saved it as a CSV file for further processing.

To prepare the data for analysis, we normalized, tokenized, and encoded it using advanced language learning models and AI techniques. This involved applying normalization methods to ensure consistency, tokenizing the text into individual words or tokens, and encoding the text using pre-trained language models. These preprocessing steps were essential for improving the accuracy and effectiveness of our recommendation engine.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/african-proverbs-recommender.git
   cd african-proverbs-recommender
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter a theme or mood in the input field.
2. Click the "Submit" button to get recommended proverbs.
3. Optionally, click the "See Another" button to see another randomly selected proverb.

## Contributing
Contributions are welcome! If you have ideas for new features, find bugs, or want to improve the application, feel free to open an issue or submit a pull request.

The recommendation engine depends on dataset quality. The current dataset for African proverbs is limited, affecting accuracy and diversity. Feel free  to expand the dataset, contribute data, curate existing content, and enhance the model. 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


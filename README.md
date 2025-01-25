# RNN IMDB Sentiment Classification

This repository contains the implementation of a Sequential Recurrent Neural Network (RNN) model for binary sentiment classification of movie reviews from the IMDB dataset. The model is designed to classify reviews as either positive or negative based on their content.

## Features
- **Deep Learning Architecture**: Includes an embedding layer, Simple RNN layer, and a Dense output layer.
- **Dataset**: Uses the IMDB dataset from TensorFlow Datasets.
- **Preprocessing**: Tokenization and padding to handle variable-length sequences.
- **Training**: Optimized the model with 1.31 million parameters for effective sequence processing.
- **Deployment**: The trained model is deployed using Streamlit for interactive user experience.

## Model Architecture
The RNN model consists of:
1. **Embedding Layer**: Converts words into dense vector representations.
2. **RNN Layer**: Processes sequential data to capture temporal dependencies.
3. **Dense Output Layer**: Outputs binary predictions (positive or negative sentiment).

**Model Summary**:
- **Vocabulary Size**: 10,000
- **Embedding Dimensions**: 128
- **RNN Units**: 128

## Requirements
To run this project, the following dependencies are required:
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Streamlit

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ghanagokul/RNN_IMDB_Classification.git
   cd RNN_IMDB_Classification
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Input a Review**:
   Enter a movie review in the provided input field. The model will classify the review as positive or negative.

## Dataset
The IMDB dataset used in this project is loaded from TensorFlow Datasets. It consists of:
- **Training Set**: 25,000 labeled reviews.
- **Test Set**: 25,000 labeled reviews.

For more details, visit [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews).

## Results
- **Accuracy**: The model achieves high accuracy on the validation set.
- **Loss**: Optimized for minimal loss during training and validation.

## Deployment
The trained model is integrated with a Streamlit application for real-time review classification. The app provides an intuitive interface for user interaction.

## Future Enhancements
- Upgrade the architecture with LSTM or GRU layers for better performance.
- Extend support for multi-class sentiment analysis.
- Add visualizations for model predictions and performance metrics.

## Repository Structure
```plaintext
RNN_IMDB_Classification/
│
├── app.py                # Streamlit app for deployment
├── model_training.py     # Script for training the RNN model
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── datasets/             # Dataset folder (if applicable)
```

## Author
**Ghana Gokul**  
- [LinkedIn](https://linkedin.com/in/ghanagokul/)  
- [GitHub](https://github.com/ghanagokul)  

If you have any questions or suggestions, feel free to reach out or open an issue in this repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

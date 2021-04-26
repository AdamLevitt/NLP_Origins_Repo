# NLP - RNN for Name Origins Prediction

### Overview
This repo analyzes the data set found at 'https://download.pytorch.org/tutorial/data.zip' and utlizes a simple ML NLP method to predict the orgins of various names. We use the Keras API to put forward a model that implements a character level embedding layer followed by a bidirectional LSTM layer and two Dense layers. 

### File Description
The files included cover the data download, the data processing, the model building and the predction against random data - brocken down as such:

    1. data_org.py:       Sets the file paths and calls download.py to execute.
    2. download.py:       Checks if the raw data or zip files exist. If required it will download/unzip the data and will store it locally. Finally it deletes the zip file.
    3. encode_func.py:    The standard sklearn encoding function.
    4. data_prep.py:      Prep the data for the model and creates a dictionary for defining the model's output.
    5. data_print:        Simple output showing the key data processing steps performed.
    6. nlp_model.py:      Build the nlp model, train the model using the data processed and store the model.
    7. call.py:           First load the model that was stored. We then use the model to define the origin of a random family name.
    8. .gitignore:        Files not shared.
  

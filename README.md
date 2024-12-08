# Bidirectional Long Short Term Memory (BiLSTM)
This repository implements a Bidirectional Long Short Term Memory (BiLSTM) model for performing Parts-of-Speech (POS) Tagging on Assamese-English code-mixed texts.

## Introduction to Parts-of-Speech (PoS) Tagging
PoS tagging is the process of identifying and labeling grammatical roles of words in texts, supporting applications like machine translation and sentiment analysis. While different languages may have their own PoS tags, I have used my own custom PoS tags for this model. The Table below defines the custom PoS tags used in this model-

![Table](https://github.com/jessicasaikia/hidden-markov-model-HMM/blob/main/Custom%20PoS%20tags%20Table.png)

## About Bidirectional Long Short Term Memory (BiLSTM) 
BiLSTM is an advanced version of the traditional LSTM network that processes data in both forward and backward directions. This dual processing allows it to capture context from both preceding and following words within a sequence, making it especially effective for tasks that rely on understanding word relationships, such as POS tagging, language modeling, and sentiment analysis.

**Algorithm**:
1.	The model imports the libraries and loads the dataset.
2.	The sentences are split into tokens (words) 
3.	POS tags are converted to numerical values. For example: AS-NOUN → 1, EN-VERB → 2
4.	Pre-trained or custom word embeddings are retrieved for each token.
5.	The sentences are padded to a fixed length.
6.	It is then processed through all five layers of model design.
7.	The output is received as a tagged sentence. 

## Where should you run this code?
I used Google Colab for this Model.
1. Create a new notebook (or file) on Google Colab.
2. Paste the code.
3. Upload your CSV dataset file to Google Colab.
4. Please make sure that you update the "path for the CSV" part of the code based on your CSV file name and file path.
5. Run the code.
6. The output will be displayed and saved as a different CSV file.

You can also VScode or any other platform (this code is just a python code)
1. In this case, you will have to make sure you have the necessary libraries installed and datasets loaded correctly.
2. Run the program for the output.
   
## Additional Notes from me
If you need any help or questions, feel free to reach out to me in the comments or via my socials. My socials are:
- Discord: jessicasaikia
- Instagram: jessicasaikiaa
- LinkedIn: jessicasaikia (www.linkedin.com/in/jessicasaikia-787a771b2)
  
Additionally, you can find the custom dictionaries that I have used in this project and the dataset in their respective repositories on my profile. Have fun coding and good luck! :D

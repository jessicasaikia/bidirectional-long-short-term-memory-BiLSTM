!pip install tensorflow numpy pandas scikit-learn

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Bidirectional

data = pd.read_csv('/content/BiLSTM.csv')

sentences = data['Sentence'].apply(lambda x: x.split())
pos_tags = data['POS_Tags'].apply(lambda x: x.split())

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(sentences)
X_seq = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in X_seq)
X_padded = pad_sequences(X_seq, padding='post', maxlen=max_length)

pos_tokenizer = Tokenizer(lower=False)
pos_tokenizer.fit_on_texts(pos_tags)
y_seq = pos_tokenizer.texts_to_sequences(pos_tags)
y_padded = pad_sequences(y_seq, padding='post', maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

y_train = np.array([np.array(i) for i in y_train])
y_test = np.array([np.array(i) for i in y_test])

embedding_dim = 100
num_pos_tags = len(pos_tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(num_pos_tags, activation='softmax')))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


def get_user_input():
    user_choice = input("Would you like to input a sentence or upload a CSV file for testing? (sentence/csv): ").strip().lower()

    if user_choice == 'sentence':
        sentence = input("Please input a sentence: ").strip()
        sentence_tokens = sentence.split()
        sentence_seq = tokenizer.texts_to_sequences([sentence_tokens])
        sentence_padded = pad_sequences(sentence_seq, padding='post', maxlen=max_length)

        prediction = model.predict(sentence_padded)
        predicted_tags = pos_tokenizer.sequences_to_texts([prediction.argmax(axis=-1)[0]])[0].split()

        print("\nSentence:", sentence)
        print("Predicted POS Tags:", predicted_tags)

    elif user_choice == 'csv':
        file_path = input("Please provide the path to the CSV file: ").strip()
        user_data = pd.read_csv(file_path)

        if 'Sentence' not in user_data.columns:
            print("CSV file must contain a 'Sentence' column.")
            return

        output_data = []
        for i, sentence in enumerate(user_data['Sentence']):
            sentence_tokens = sentence.split()
            sentence_seq = tokenizer.texts_to_sequences([sentence_tokens])
            sentence_padded = pad_sequences(sentence_seq, padding='post', maxlen=max_length)

            prediction = model.predict(sentence_padded)
            predicted_tags = pos_tokenizer.sequences_to_texts([prediction.argmax(axis=-1)[0]])[0].split()

            output_data.append({
                "Sentence": sentence,
                "Predicted POS Tags": predicted_tags
            })

        output_df = pd.DataFrame(output_data)
        output_df.to_csv('BiLSTMoutput.csv', index=False)
        print("Output is saved as 'BiLSTMoutput.csv'.")

    else:
        print("Invalid input. Please choose either 'sentence' or 'csv'.")
        get_user_input()


get_user_input()

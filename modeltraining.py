# import nltk
# # nltk.download('all')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import json
# import pickle
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
# import random
import warnings
# import tensorflow
warnings.filterwarnings('ignore')

# words=[]
# classes = []
# documents = []
# ignore_words = ['?', '!']
# data_file = open('intents.json').read()
# intents = json.loads(data_file)

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         documents.append((w, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))
# classes = sorted(list(set(classes)))
# print (len(documents), "documents")
# print (len(classes), "classes", classes)
# print (len(words), "unique lemmatized words", words)

# # pickle.dump(words,open('words.pkl','wb'))
# # pickle.dump(classes,open('classes.pkl','wb'))

# training = []
# output_empty = [0] * len(classes)
# for doc in documents:
#     bag = []
#     pattern_words = doc[0]
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])
# random.shuffle(training)
# training = np.array(training)
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print("Training data created")

# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


###################################################################################################


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
# from spellchecker import SpellChecker
# spell_checker = SpellChecker()

data_file = open('intents.json').read()
intents = json.loads(data_file)

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

model = load_model("chat_model.h5")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")

    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25  
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        intent = {"intent": classes[r[0]], "probability": str(r[1])}

        for intent_data in intents['intents']:
            if intent_data['tag'] == intent['intent']:
                responses = intent_data.get('responses', [])
                links = intent_data.get('links', [])
                intent['response'] = random.choice(responses) if responses else ""
                intent['links'] = links
                break

        return_list.append(intent)

    return return_list


while True:
    input_sentence = input(">>>> ")
    if input_sentence.lower() == "bye":
        break
    else:
        # input_sentence = ' '.join([spell_checker.correction(word) for word in nltk.word_tokenize(input_sentence)])
        predictions = predict_class(input_sentence, model)
        high_prob_prediction = next((p for p in predictions if float(p["probability"]) > 0.7), None)

        if high_prob_prediction:
            print("Intent:", high_prob_prediction["intent"])
            print("Probability:", high_prob_prediction["probability"])
            print("Response:", high_prob_prediction["response"])
            print("Links:", high_prob_prediction["links"])
        else:
            print("No high probability intent found.")

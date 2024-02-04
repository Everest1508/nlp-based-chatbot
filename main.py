from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler, CallbackQueryHandler
from telegram import ParseMode
from creds import Key
CHOOSING, NAME, COLLEGE, PERSONA = range(4)
ANSWER=0
TOKEN = Key
from  databaseconn import save_to_db,find_user_details,save_history
import logging
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# # # # NLP # # # #

data_file = open('intents.json').read()
intents = json.loads(data_file)

with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

model = load_model("chat_model.h5")
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


def give_answer(input_sentence):
    predictions = predict_class(input_sentence, model)
    high_prob_prediction = next((p for p in predictions if float(p["probability"]) > 0.7), None)

    if high_prob_prediction:
        return high_prob_prediction
    else:
        return {"intent":"unknown","probability":1.0,"response":"Please Rephrase your question!","links":[]}


# # # # Registration Part # # # #


def start(update: Update, context: CallbackContext) -> int:
    if context.user_data:
        update.message.reply_text(f"Welcome Back {context.user_data['name']}!")
        update.message.reply_text('''I am here to assist you. How can we help you today?''')
        return ConversationHandler.END
    elif find_user_details(str(update.message.from_user.id)):
        info = find_user_details(str(update.message.from_user.id))
        param_list = ['id','name','college','persona']
        param_list_php = ['ID', 'Name','College Name','Persona']
        for i in range(4):
            context.user_data[param_list[i]]=info[param_list_php[i]]

        update.message.reply_text(f"Welcome Back {context.user_data['name']}!")
        update.message.reply_text('''I am here to assist you. How can we help you today?''')
        return ConversationHandler.END
    update.message.reply_text("Hi! I'm your Cognifront bot. What's your name?")
    return NAME

def get_name(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    context.user_data['id'] = str(user.id)
    print(user.id)
    context.user_data['name'] = update.message.text
    update.message.reply_text(f"Nice to meet you, {context.user_data['name']}! What's the name of your college?")
    return COLLEGE

def get_college(update: Update, context: CallbackContext) -> int:
    context.user_data['college'] = update.message.text
    update.message.reply_text("Great! Which persona do you identify with?")

    keyboard = [
        [InlineKeyboardButton("First Year Engineering Student", callback_data='First Year Engineering Student')],
        [InlineKeyboardButton("Third Year Engineering Student", callback_data='Third Year Engineering Student')],
        [InlineKeyboardButton("Fourth Year Engineering Student", callback_data='Fourth Year Engineering Student')],
        [InlineKeyboardButton("MCA Student", callback_data='MCA Student')],
        [InlineKeyboardButton("Diploma Student", callback_data='Diploma Student')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Please choose your persona:', reply_markup=reply_markup)

    return PERSONA


def get_persona(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    
    context.user_data['persona'] = query.data
    query.answer()

    # Save registration details to DB
    save_to_db(context.user_data)

    query.message.reply_text('''I am here to assist you. How can we help you today?''')

    return ConversationHandler.END



def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Registration canceled.")
    return ConversationHandler.END

# # # # End of Registration # # # #

# # # # Sending Attachments to the user # # # #


def interns(update:Update,context:CallbackContext)->None:
    update.message.reply_text("Our interns")
    update.message.reply_video(video="https://databasetelegram.000webhostapp.com/Videos/intern1.mp4")
    update.message.reply_video(video="https://databasetelegram.000webhostapp.com/Videos/intern2.mp4")
    update.message.reply_video(video="https://databasetelegram.000webhostapp.com/Videos/intern4.mp4")
    update.message.reply_video(video="https://databasetelegram.000webhostapp.com/Videos/intern5.mp4")

def handbook(update:Update,context:CallbackContext):
    update.message.reply_text("Interns Handbook")
    update.message.reply_document(document="https://databasetelegram.000webhostapp.com/Videos/interns-handbook.pdf")

def messages(update:Update,context:CallbackContext):
    if context.user_data:
        
        text = update.message.text
        response = give_answer(text)
        update.message.reply_text(response['response'],parse_mode=ParseMode.HTML)
        if len(response['links'])!=0:
            for i in response['links']:
                if ".mp4" in i:
                    update.message.reply_video(video=i)
                elif ".pdf" in i:
                    update.message.reply_document(document=i)
                else:
                    update.message.reply_text(i)
        answer = 0 if response['intent']=="unknown" else 1
        save_history({"id":context.user_data['id'],"message":update.message.text,"answered":answer})

    
    else:
        update.message.reply_text("Prior to proceeding, kindly execute the /start command.")
   

def main() -> None:
    updater = Updater(TOKEN)

    dp = updater.dispatcher

    # Conversation states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            NAME: [MessageHandler(Filters.text & ~Filters.command, get_name)],
            COLLEGE: [MessageHandler(Filters.text & ~Filters.command, get_college)],
            PERSONA: [CallbackQueryHandler(get_persona)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    # Handlers
    dp.add_handler(conv_handler)
    dp.add_handler(CommandHandler('handbook',handbook))
    dp.add_handler(CommandHandler('interns',interns))
    dp.add_handler(MessageHandler(Filters.text,messages))
    
    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

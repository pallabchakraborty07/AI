import nltk
from nltk.chat.util import Chat,reflections

reflections = {
    "I am":"You are",
    "I am":"You are",
    "You are":"I am",
    "I have": "You have",
    "You have":"I have",
    "Me":"You",
    "You": "Me",
    "I was": "You were",
    "You were":"I was",
    "I will":"You will"
}

pairs = [
    [
        r"hi|hello|hey",
        ["hey !how i can help you"]
    ],
    r"hi hello hey",

    [
        "Hey! How I can help you today?",
    ],
    r"What is your name ",
    [
    "My Name is Codingal Jarvis Created by Atiya & Pallab",
    ],
    r"How are you",

    [
        "I'm Doing Good. What About You ?",
    ],
        r"I'm Fine",
]
def start():
    print("hello , my name is jarvis")
    chat = Chat(pairs,reflections)
    chat.converse()

start()
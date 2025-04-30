# speaker.py
import sys
import pyttsx3

text = sys.argv[1]

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.say(text)
engine.runAndWait()
engine.stop()

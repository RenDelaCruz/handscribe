from sign_language_translator import SignLanguageTranslator

if __name__ == "__main__":
    translator = SignLanguageTranslator(show_landmarks=False, show_bounding_rectangle=False)
    translator.start()

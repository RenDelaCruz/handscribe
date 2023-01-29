from sign_language_translator import SignLanguageTranslator

if __name__ == "__main__":
    translator = SignLanguageTranslator(
        show_landmarks=False,
        show_bounding_box=True,
    )
    translator.start()

# cv2.circle(
#     image,
#     (
#         (bounding_box.x + bounding_box.x2) // 2,
#         (bounding_box.y + bounding_box.y2) // 2,
#     ),
#     max(
#         bounding_box.y2 - bounding_box.y,
#         bounding_box.x2 - bounding_box.x,
#     )
#     // 2 + self.padding,
#     Colour.CYAN.value,
#     2,
# )

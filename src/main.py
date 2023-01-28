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
#         (bounding_box.x + bounding_box.width) // 2,
#         (bounding_box.y + bounding_box.height) // 2,
#     ),
#     max(
#         bounding_box.height - bounding_box.y,
#         bounding_box.width - bounding_box.x,
#     )
#     // 2,
#     Colour.CYAN.value,
#     2,
# )

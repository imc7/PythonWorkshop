import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:
    image=cv2.imread('files/hands.png')
    heigth, width, _=image.shape
    print('Altura: ', heigth)
    print('Ancho: ', width)
    image=cv2.resize(image, (3*width,3*heigth))

    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result=hands.process(image_rgb)

    if result.multi_hand_landmarks is not None:
        for mano in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, mano, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(51, 97, 255), thickness=4, circle_radius=5),
                mp_drawing.DrawingSpec(color=(122, 255, 51), thickness=4)
            )

    cv2.imshow("Hands image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

            
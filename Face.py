import cv2
import mediapipe as mp


mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    min_detection_confidence=0.5
) as face_mesh:
    image=cv2.imread('files/face2.jpg')

    # Redimensionar imágenes #########################
    height, width, _ = image.shape
    print("Altura: {}".format(height))
    print("Ancho: {}".format(width))

    image=cv2.resize(image, (3*width, 3*height))

    height, width, _ = image.shape
    print("Altura nueva: {}".format(height))
    print("Ancho nuevo: {}".format(width))

    ###################################################

    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result=face_mesh.process(image_rgb)
    detection=result.multi_face_landmarks

    cont=0
    if detection is not None:
        for face_landmarks in detection:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            cont=cont+1

    text="Rostros: {}".format(cont)
    cv2.putText(image,text , (10, 40), 1, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Face', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
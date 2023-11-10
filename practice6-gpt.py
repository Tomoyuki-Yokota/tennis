import streamlit as st
import mediapipe as mp 
import cv2
import tempfile

st.title('テニスフォーム判別アプリ')
st.write('Play Tennis')

uploaded_file = st.file_uploader('Choose a video...', type='mp4')
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0,255,0))
    mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,0,255))

    cap_file = cv2.VideoCapture(tfile.name)
    frames = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic_detection:
        while cap_file.isOpened():
            success, image = cap_file.read()
            if not success:
                break
            image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic_detection.process(rgb_image)
            
            # ... (landmarks drawing code remains unchanged)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.face_landmarks,
connections=mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=None,
connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.pose_landmarks,
    connections=mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,
    connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.left_hand_landmarks,
    connections=mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.right_hand_landmarks,
    connections=mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)

            frames.append(image)

    cap_file.release()

    # Convert frames to video for display in Streamlit
    h, w, layers = frames[0].shape
    size = (w, h)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

    video_file = open('output.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

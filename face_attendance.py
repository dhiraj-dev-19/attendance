import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import datetime

# Step 1: Load known faces
path = 'known_faces'
images = [] 
names = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    names.append(os.path.splitext(file)[0])

# Step 2: Encode known faces
def findEncodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)
        if encode:  # only if a face is detected
            encode_list.append(encode[0])
    return encode_list

known_encodings = findEncodings(images)



# Step 3: Mark attendance
# Dictionary to store the last attendance date for each person
last_attended = {}

# To track if attendance for a person has been marked during the current session
attendance_marked_today = {}

def markAttendance(name):
    # Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime('%Y-%m-%d')

    # Check if the CSV file exists
    file_exists = os.path.exists('attendance.csv')

    # Check if this person was already marked today
    if name in last_attended and last_attended[name] == today:
        # If attendance has already been marked today, skip
        if name not in attendance_marked_today:
            print(f"{name} has already been marked for today.")
        return  # Skip marking attendance if already marked today

    # If the file exists, check if the person's name and today's date already exist
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        lines = f.readlines()

        # Check if the person has already been recorded for today in the CSV
        if any(f"{name},{today}" in line for line in lines):
            if name not in attendance_marked_today:
                print(f"{name} has already been marked for today.")
            return  # Skip if already recorded for today

        # Otherwise, record the attendance with the current time
        time_str = datetime.now().strftime('%H:%M:%S')
        f.write(f'{name},{today},{time_str}\n')

        # Update the last attended date for this person
        last_attended[name] = today

        # Mark that attendance has been marked for this person today
        attendance_marked_today[name] = True

        # Print the message only once
        print(f"Attendance marked for {name} on {today} at {time_str}")
# Step 4: Start webcam and detect faces
cap = cv2.VideoCapture(0)

print("📷 Press space  to quit webcam.")

while True:
    success, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(rgb_small)
    encodes_current_frame = face_recognition.face_encodings(rgb_small, faces_current_frame)

    for encode_face, face_loc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_encodings, encode_face)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = names[best_match_index].capitalize()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

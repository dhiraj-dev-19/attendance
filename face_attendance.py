import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.known_encodings = []
        self.names = []
        self.cap = None
        self.running = False
        
        # Create GUI components
        self.create_widgets()
        self.load_known_faces()
        
    def create_widgets(self):
        # Webcam display
        self.video_frame = ttk.LabelFrame(self.root, text="Live Webcam Feed")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(padx=10, pady=10)
        
        # System messages
        self.msg_frame = ttk.LabelFrame(self.root, text="System Messages")
        self.msg_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.message_area = scrolledtext.ScrolledText(self.msg_frame, width=50, height=20)
        self.message_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control buttons
        self.btn_frame = ttk.Frame(self.msg_frame)
        self.btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(self.btn_frame, text="Start", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(self.btn_frame, text="Exit", command=self.cleanup)
        self.exit_btn.pack(side=tk.LEFT, padx=5)
        
    def log_message(self, message):
        self.message_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.message_area.see(tk.END)
        
    def load_known_faces(self):
        known_faces_dir = 'known_faces'
        images = []
        self.names = []
        
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            self.log_message(f"Created '{known_faces_dir}' directory. Add face images there.")
            return
            
        for file in os.listdir(known_faces_dir):
            img_path = os.path.join(known_faces_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                self.names.append(os.path.splitext(file)[0])
        
        if images:
            self.known_encodings = self.find_encodings(images)
            self.log_message(f"Loaded {len(self.names)} known faces")
        else:
            self.log_message("No valid face images found in 'known_faces' folder")
            
    def find_encodings(self, images):
        encode_list = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img_rgb)
            if encode:
                encode_list.append(encode[0])
            else:
                self.log_message("Warning: No face detected in an image")
        return encode_list
        
    def start_system(self):
        if not self.known_encodings:
            messagebox.showerror("Error", "No known faces loaded. Add images to 'known_faces' folder.")
            return
            
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
            
        self.running = True
        self.start_btn.config(text="Stop", command=self.stop_system)
        self.log_message("System started - Showing live feed")
        self.update_frame()
        
    def stop_system(self):
        self.running = False
        self.start_btn.config(text="Start", command=self.start_system)
        self.log_message("System paused")
        
    def update_frame(self):
        if self.running:
            success, frame = self.cap.read()
            if success:
                # Process frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                
                for encode_face, face_loc in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(self.known_encodings, encode_face)
                    face_distances = face_recognition.face_distance(self.known_encodings, encode_face)
                    
                    if len(face_distances) > 0:
                        best_match_idx = np.argmin(face_distances)
                        if matches[best_match_idx]:
                            name = self.names[best_match_idx].capitalize()
                            self.mark_attendance(name)
                            
                            # Draw bounding box
                            y1, x2, y2, x1 = face_loc
                            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x1+6, y2-6), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Convert to Tkinter format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            self.root.after(30, self.update_frame)
            
    def mark_attendance(self, name):
        try:
            attendance_file = str(Path.home() / "Documents" / "face_attendance.csv")
            today = datetime.now().strftime('%Y-%m-%d')
            
            if os.path.exists(attendance_file):
                with open(attendance_file, 'r') as f:
                    if any(f"{name},{today}" in line for line in f):
                        self.log_message(f"{name}: Attendance already marked")
                        return
                        
            with open(attendance_file, 'a') as f:
                time_str = datetime.now().strftime('%H:%M:%S')
                f.write(f'{name},{today},{time_str}\n')
                self.log_message(f"{name}: Attendance marked at {time_str}")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            
    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import threading

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
        self.latest_frame = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.display_width = 640
        self.display_height = 480
        
        # Create GUI components
        self.create_widgets()
        self.load_known_faces()
        
    def create_widgets(self):
        # Webcam display frame
        self.video_frame = ttk.LabelFrame(self.root, text="Live Webcam Feed")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(padx=10, pady=10)

        # System messages frame
        self.msg_frame = ttk.LabelFrame(self.root, text="System Controls")
        self.msg_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Message area
        self.message_area = scrolledtext.ScrolledText(self.msg_frame, width=50, height=20)
        self.message_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control buttons
        self.create_control_buttons()

    def create_control_buttons(self):
        btn_frame = ttk.Frame(self.msg_frame)
        btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(btn_frame, text="Exit", command=self.cleanup)
        self.exit_btn.pack(side=tk.LEFT, padx=5)
        
        self.add_face_btn = ttk.Button(
            btn_frame, 
            text="Register Face", 
            command=self.register_new_face
        )
        self.add_face_btn.pack(side=tk.LEFT, padx=5)

    def log_message(self, message):
        self.message_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.message_area.see(tk.END)
        
    def load_known_faces(self):
        known_faces_dir = Path('known_faces')
        images = []
        self.names = []
        
        try:
            known_faces_dir.mkdir(exist_ok=True)
            
            for file in known_faces_dir.glob("*.*"):
                img = cv2.imread(str(file))
                if img is not None:
                    images.append(img)
                    self.names.append(file.stem)
            
            if images:
                self.known_encodings = self.find_encodings(images)
                self.log_message(f"Loaded {len(self.names)} known faces")
            else:
                self.log_message("No faces found in 'known_faces' directory")
                
        except Exception as e:
            self.log_message(f"Error loading faces: {str(e)}")

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
            messagebox.showerror("Error", "No known faces loaded. Register faces first.")
            return
            
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            self.running = True
            self.start_btn.config(text="Stop", command=self.stop_system)
            self.log_message("System started - Live feed active")
            
            # Start threads
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.capture_thread.start()
            self.processing_thread.start()
            
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.cleanup()

    def capture_frames(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.latest_frame = cv2.resize(frame, (self.display_width, self.display_height))

    def process_frames(self):
        while self.running:
            if self.latest_frame is None:
                continue
                
            with self.lock:
                frame = self.latest_frame.copy()
            
            # Process at lower resolution
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            # Recognition
            for encode_face, face_loc in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_encodings, encode_face)
                face_distances = face_recognition.face_distance(self.known_encodings, encode_face)
                
                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = self.names[best_match_idx].capitalize()
                        self.mark_attendance(name)
                        
                        # Draw annotations
                        y1, x2, y2, x1 = face_loc
                        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1+6, y2-6), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            with self.lock:
                self.processed_frame = frame

    def update_frame(self):
        if self.running and self.processed_frame is not None:
            with self.lock:
                frame = self.processed_frame.copy()
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(30, self.update_frame)

    def register_new_face(self):
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
        
        name = simpledialog.askstring("Face Registration", "Enter person's name:")
        if not name:
            messagebox.showwarning("Cancelled", "Registration cancelled")
            return
            
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Invalid image file")
            
            # Face validation
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                raise ValueError("No face detected in image")
            if len(face_locations) > 1:
                raise ValueError("Multiple faces detected")
            
            # Save to known_faces
            save_dir = Path('known_faces')
            save_dir.mkdir(exist_ok=True)
            
            if any(save_dir.glob(f"{name}.*")):
                raise ValueError(f"Name '{name}' already exists")
            
            cv2.imwrite(str(save_dir / f"{name}.jpg"), img)
            self.load_known_faces()
            messagebox.showinfo("Success", f"{name} registered successfully")
            
        except Exception as e:
            messagebox.showerror("Registration Error", str(e))

    def mark_attendance(self, name):
        try:
            attendance_file = Path.home() / "Documents" / "attendance.csv"
            today = datetime.now().strftime('%Y-%m-%d')
            time_str = datetime.now().strftime('%H:%M:%S')
            
            # Check existing entries
            if attendance_file.exists():
                with open(attendance_file, 'r') as f:
                    if any(f"{name},{today}" in line for line in f):
                        self.log_message(f"{name}: Already marked")
                        return
            
            # Write new entry
            with open(attendance_file, 'a') as f:
                f.write(f"{name},{today},{time_str}\n")
                self.log_message(f"{name}: Attendance recorded at {time_str}")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")

    def stop_system(self):
        self.running = False
        self.start_btn.config(text="Start", command=self.start_system)
        self.log_message("System paused")

    # Add the missing cleanup method
    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import os

class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detector (PT & ONNX)")
        self.root.geometry("1400x800")
        
        self.model = None
        self.current_image = None
        self.camera = None
        self.camera_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame - Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Model Loading
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(control_frame, text="No model loaded", foreground="red")
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Detection Options
        ttk.Button(control_frame, text="Detect Image", command=self.detect_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Detect Folder", command=self.detect_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Live Camera", command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Export to ONNX
        ttk.Button(control_frame, text="Export to ONNX", command=self.export_to_onnx).pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Clear Button
        ttk.Button(control_frame, text="Clear", command=self.clear_images).pack(side=tk.LEFT, padx=5)
        
        # Confidence Threshold
        ttk.Label(control_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_spinbox = ttk.Spinbox(control_frame, from_=0.1, to=1.0, increment=0.05, 
                                   textvariable=self.conf_var, width=10)
        conf_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Main Frame - Image Display
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Input Image
        left_frame = ttk.LabelFrame(main_frame, text="Input Image", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.input_canvas = tk.Canvas(left_frame, bg="gray20", highlightthickness=0)
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right Panel - Output Image
        right_frame = ttk.LabelFrame(main_frame, text="Detection Result", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.output_canvas = tk.Canvas(right_frame, bg="gray20", highlightthickness=0)
        self.output_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def load_model(self):
        """Load YOLO model from PT or ONNX file"""
        file_path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("Model files", "*.pt *.onnx"), ("PT files", "*.pt"), 
                      ("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading model...")
                self.root.update()
                
                # Ultralytics YOLO có thể load cả PT và ONNX
                self.model = YOLO(file_path)
                
                model_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.onnx':
                    self.model_label.config(text=f"ONNX: {model_name}", foreground="blue")
                else:
                    self.model_label.config(text=f"YOLO: {model_name}", foreground="green")
                
                self.status_var.set(f"Model loaded: {model_name}")
                messagebox.showinfo("Success", f"Model loaded successfully!\n{model_name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.status_var.set("Error loading model")
                
    def export_to_onnx(self):
        """Export current PT model to ONNX format"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a PT model first!")
            return
        
        # Check if current model is already ONNX
        if hasattr(self.model, 'model_name') and self.model.model_name.endswith('.onnx'):
            messagebox.showinfo("Info", "Current model is already in ONNX format!")
            return
        
        try:
            self.status_var.set("Exporting to ONNX...")
            self.root.update()
            
            # Export to ONNX
            output_path = self.model.export(format="onnx")
            
            messagebox.showinfo("Success", f"Model exported to ONNX!\n{output_path}")
            self.status_var.set(f"Exported to: {output_path}")
            
            # Ask if user wants to load the ONNX model
            if messagebox.askyesno("Load ONNX", "Do you want to load the exported ONNX model?"):
                self.model = YOLO(output_path)
                model_name = os.path.basename(output_path)
                self.model_label.config(text=f"ONNX: {model_name}", foreground="blue")
                self.status_var.set(f"Model loaded: {model_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export model:\n{str(e)}")
            self.status_var.set("Error exporting model")
                
    def detect_image(self):
        """Detect objects in a single image"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("Processing image...")
                self.root.update()
                
                # Read and display input image
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(image_rgb, self.input_canvas)
                
                # Run detection - YOLO tự động xử lý cả PT và ONNX
                results = self.model(file_path, conf=self.conf_var.get())
                
                # Get annotated image
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                self.display_image(annotated_rgb, self.output_canvas)
                
                # Count detections
                num_detections = len(results[0].boxes)
                self.status_var.set(f"Detection complete - {num_detections} objects detected")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
                self.status_var.set("Error processing image")
                
    def detect_folder(self):
        """Detect objects in all images in a folder"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        
        if folder_path:
            try:
                # Get all image files
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                image_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(image_extensions)]
                
                if not image_files:
                    messagebox.showinfo("Info", "No image files found in folder")
                    return
                
                self.status_var.set(f"Processing {len(image_files)} images...")
                self.root.update()
                
                # Process first image and display
                first_image_path = os.path.join(folder_path, image_files[0])
                image = cv2.imread(first_image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(image_rgb, self.input_canvas)
                
                results = self.model(first_image_path, conf=self.conf_var.get())
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                self.display_image(annotated_rgb, self.output_canvas)
                
                # Ask to save all results
                if messagebox.askyesno("Save Results", 
                    f"Process all {len(image_files)} images and save results?"):
                    
                    output_folder = os.path.join(folder_path, "detections")
                    os.makedirs(output_folder, exist_ok=True)
                    
                    for i, img_file in enumerate(image_files):
                        img_path = os.path.join(folder_path, img_file)
                        results = self.model(img_path, conf=self.conf_var.get())
                        
                        # Save annotated image
                        output_path = os.path.join(output_folder, f"detected_{img_file}")
                        results[0].save(output_path)
                        
                        self.status_var.set(f"Processing {i+1}/{len(image_files)}...")
                        self.root.update()
                    
                    messagebox.showinfo("Success", 
                        f"Processed {len(image_files)} images!\nSaved to: {output_folder}")
                    
                self.status_var.set(f"Folder processing complete - {len(image_files)} images")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process folder:\n{str(e)}")
                self.status_var.set("Error processing folder")
                
    def toggle_camera(self):
        """Toggle live camera detection"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Start live camera detection"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot access camera!")
                return
                
            self.camera_running = True
            self.status_var.set("Camera running - Press 'Live Camera' again to stop")
            
            # Start camera thread
            threading.Thread(target=self.camera_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
            
    def stop_camera(self):
        """Stop live camera detection"""
        self.camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.status_var.set("Camera stopped")
        
    def camera_loop(self):
        """Main camera processing loop"""
        while self.camera_running:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Display input frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(frame_rgb, self.input_canvas)
            
            # Run detection - works with both PT and ONNX
            results = self.model(frame, conf=self.conf_var.get())
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Display output frame
            self.display_image(annotated_rgb, self.output_canvas)
            
    def display_image(self, image, canvas):
        """Display image on canvas with proper scaling"""
        if image is None:
            return
            
        # Get canvas size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep reference
        
    def clear_images(self):
        """Clear both canvas panels"""
        self.input_canvas.delete("all")
        self.output_canvas.delete("all")
        self.status_var.set("Images cleared")
        
    def on_closing(self):
        """Handle window closing"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
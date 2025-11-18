import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import requests
import io
import threading
import os

class TICDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("TIC Orthofoto Building Detector")
        self.root.geometry("1400x900")
        
        self.current_image = None
        self.result_image = None
        self.clf = None
        self.scaler = None
        
        # Load ML model
        self.load_model()
        
        # Build GUI
        self.build_gui()
        
        self.log("="*80)
        self.log("🛰️ TIC ORTHOFOTO + ML BUILDING DETECTOR")
        self.log("="*80)
        self.log("✅ System ready")
        if os.environ.get('API_KEY'):
            self.log("� API key loaded from environment variable")
        else:
            self.log("📝 Please enter your TIC API key")
        self.log("")
    
    def build_gui(self):
        """Build the GUI"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="🛰️ TIC Orthofoto Building Detector", 
                         font=('Arial', 18, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left panel
        control_frame = ttk.LabelFrame(main_frame, text="🎯 Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # API Key input
        api_frame = ttk.LabelFrame(control_frame, text="🔑 TIC API Key", padding="10")
        api_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        # Load API key from environment variable if available
        env_api_key = os.environ.get('API_KEY', '')
        self.api_key_var = tk.StringVar(value=env_api_key)
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=40, show="*")
        api_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        api_frame.columnconfigure(1, weight=1)
        
        # Coordinate input
        coord_frame = ttk.LabelFrame(control_frame, text="📍 Coordinates", padding="10")
        coord_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coord_frame, text="Latitude:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.lat_var = tk.StringVar(value="57.48484687")
        ttk.Entry(coord_frame, textvariable=self.lat_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(coord_frame, text="Longitude:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lon_var = tk.StringVar(value="15.02520248")
        ttk.Entry(coord_frame, textvariable=self.lon_var, width=20).grid(row=1, column=1, padx=5)
        
        # Quick presets
        preset_frame = ttk.LabelFrame(control_frame, text="📌 Quick Locations", padding="5")
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="Stockholm", 
                  command=lambda: self.set_location(59.3293, 18.0686)).pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame, text="Göteborg", 
                  command=lambda: self.set_location(57.7089, 11.9746)).pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame, text="Malmö", 
                  command=lambda: self.set_location(55.6050, 13.0038)).pack(fill=tk.X, pady=2)
        
        # Detection Mode
        mode_frame = ttk.LabelFrame(control_frame, text="🎯 Detection Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.detection_mode = tk.StringVar(value="solar")
        ttk.Radiobutton(mode_frame, text="☀️ Solar Panels", variable=self.detection_mode, 
                       value="solar", command=self.on_mode_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(mode_frame, text="🏠 Buildings", variable=self.detection_mode, 
                       value="buildings", command=self.on_mode_change).pack(anchor=tk.W, pady=2)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(control_frame, text="⚙️ Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Min Confidence:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.35)
        self.confidence_slider = ttk.Scale(settings_frame, from_=0.2, to=0.9, 
                                          variable=self.confidence_var, orient=tk.HORIZONTAL,
                                          command=self.on_confidence_change)
        self.confidence_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.confidence_label = ttk.Label(settings_frame, text="35%")
        self.confidence_label.grid(row=0, column=2)
        settings_frame.columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.fetch_btn = ttk.Button(button_frame, text="🚀 Fetch & Analyze", 
                                    command=self.fetch_and_analyze)
        self.fetch_btn.pack(fill=tk.X, pady=5)
        
        self.detect_btn = ttk.Button(button_frame, text="☀️ Detect Solar Panels", 
                                     command=self.start_detection, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=5)
        
        # Status
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Console
        console_frame = ttk.LabelFrame(control_frame, text="📋 Console", padding="5")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=15, width=40, 
                                                 wrap=tk.WORD, font=('Consolas', 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Display
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
        
        # Notebook
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="📷 Original")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg='black')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="☀️ Detections")
        
        self.results_canvas = tk.Canvas(self.results_frame, bg='black')
        self.results_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Stats
        stats_frame = ttk.LabelFrame(display_frame, text="📊 Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="No detections yet", font=('Arial', 10))
        self.stats_label.pack()
    
    def log(self, message):
        """Log to console"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.update()
    
    def load_model(self):
        """Load ML model"""
        try:
            if os.path.exists('building_classifier.pkl'):
                with open('building_classifier.pkl', 'rb') as f:
                    self.clf, self.scaler = pickle.load(f)
                return True
            return False
        except:
            return False
    
    def on_confidence_change(self, value):
        """Update confidence label"""
        conf = float(value)
        self.confidence_label.config(text=f"{conf*100:.0f}%")
    
    def on_mode_change(self):
        """Update UI when detection mode changes"""
        mode = self.detection_mode.get()
        if mode == "solar":
            self.detect_btn.config(text="☀️ Detect Solar Panels")
            self.log("🔄 Mode: Solar Panel Detection")
        else:
            self.detect_btn.config(text="🏠 Detect Buildings")
            self.log("🔄 Mode: Building Detection")
    
    def set_location(self, lat, lon):
        """Set location"""
        self.lat_var.set(str(lat))
        self.lon_var.set(str(lon))
        self.log(f"📍 Location set to: {lat}, {lon}")
    
    def query_tic_api(self, lat, lon, api_key):
        """Query TIC API to get orthofoto URL"""
        try:
            self.log("🔍 Querying TIC API for orthofoto...")
            self.log(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
            self.log(f"📍 Coordinates: lat={lat}, lon={lon}")
            self.log("")
            
            # Use the CORRECT TIC API format from documentation
            api_url = "https://api.tic.io/datasets/properties/se/orthophotos"
            
            headers = {
                'x-api-key': api_key,
                'Accept': 'application/json'
            }
            
            # TIC API uses 'latitude' and 'bufferDistance' parameters
            params = {
                'latitude': lat,
                'longitude': lon,
                'bufferDistance': 200  # 200 meters buffer
            }
            
            self.log(f"📡 GET {api_url}")
            self.log(f"   Params: latitude={lat}, longitude={lon}, bufferDistance=200")
            self.log("")
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            
            self.log(f"📊 Response status: {response.status_code}")
            
            if response.status_code != 200:
                self.log(f"❌ API error: {response.text[:300]}")
                return None
            
            self.log("✅ API request successful!")
            self.log("")
            
            # Parse the response
            data = response.json()
            self.log(f"✅ API response received")
            
            # TIC API returns a LIST of orthophotos for different years
            if isinstance(data, list) and len(data) > 0:
                self.log(f"📦 Found {len(data)} orthophotos for this location")
                self.log("")
                
                # Show available years
                for item in data:
                    year = item.get('flightYear', 'unknown')
                    url = item.get('url', '')
                    res = item.get('resolution', 'unknown')
                    self.log(f"   Year {year}: resolution {res}m - {url[:60]}...")
                
                self.log("")
                
                # Get the LATEST (most recent) orthofoto
                latest = max(data, key=lambda x: x.get('flightYear', 0))
                url = latest.get('url')
                year = latest.get('flightYear')
                
                if url:
                    self.log(f"✅ Using latest orthofoto: Year {year}")
                    self.log(f"   URL: {url}")
                    return url
                else:
                    self.log(f"❌ No URL found in latest orthofoto")
                    return None
            
            elif isinstance(data, dict) and 'url' in data:
                # Single orthofoto response
                return data['url']
            
            else:
                self.log(f"⚠️ Unexpected response format")
                self.log(f"   Type: {type(data)}")
                self.log(f"   Data: {str(data)[:200]}")
                return None
                
        except Exception as e:
            self.log(f"❌ API error: {str(e)}")
            return None
    
    def fetch_and_analyze(self):
        """Fetch and analyze"""
        self.fetch_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Fetching...", foreground="orange")
        
        thread = threading.Thread(target=self.fetch_orthofoto)
        thread.daemon = True
        thread.start()
    
    def fetch_orthofoto(self):
        """Fetch from coordinates"""
        try:
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())
            
            self.log(f"\n{'='*60}")
            self.log("📥 FETCHING ORTHOFOTO FROM TIC")
            self.log(f"{'='*60}")
            self.log(f"📍 Coordinates: {lat}, {lon}")
            self.log("")
            
            # api_key = self.api_key_var.get().strip()
            load_dotenv()
            api_key = os.environ.get('API_KEY', '').strip()
            
            if not api_key:
                raise Exception("Please enter your TIC API key.\nYou can get one from TIC.io")
            
            # Query TIC API for orthofoto URL
            url = self.query_tic_api(lat, lon, api_key)
            
            if not url:
                raise Exception("Could not get orthofoto URL from TIC API.\nCheck your API key and coordinates.")
            
            self.log("")
            self.log(f"📥 Downloading from: {url[:80]}...")
            
            # Add authorization header for download
            headers = {
                'Authorization': f'Bearer {api_key}'
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                self.log(f"✅ Downloaded {len(response.content)} bytes")
                
                # Convert to array
                img_pil = Image.open(io.BytesIO(response.content))
                img_array = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
                self.log(f"✅ Image size: {img_array.shape[1]}×{img_array.shape[0]} pixels")
                
                self.current_image = img_array
                cv2.imwrite('temp_orthofoto.png', img_array)
                
                # Update GUI
                self.root.after(0, self.display_original)
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
                self.root.after(0, lambda: self.fetch_btn.config(state=tk.NORMAL))
                
                # Auto-detect
                self.root.after(1000, self.start_detection)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(text="Error", foreground="red"))
            self.root.after(0, lambda: self.fetch_btn.config(state=tk.NORMAL))
    
    def display_original(self):
        """Display original"""
        if self.current_image is None:
            return
        
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.original_photo = ImageTk.PhotoImage(img_pil)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(canvas_width//2, canvas_height//2, 
                                         image=self.original_photo, anchor=tk.CENTER)
    
    def start_detection(self):
        """Start detection"""
        if self.current_image is None:
            return
        
        self.detect_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Detecting...", foreground="orange")
        
        thread = threading.Thread(target=self.detect_buildings)
        thread.daemon = True
        thread.start()
    
    def detect_buildings(self):
        """Detect buildings or solar panels based on mode"""
        try:
            mode = self.detection_mode.get()
            
            if mode == "solar":
                self.log(f"\n{'='*60}")
                self.log("☀️ DETECTING SOLAR PANELS (Frame-Based V2)")
                self.log(f"{'='*60}")
                
                from detect_solar_panels_v2 import detect_solar_panels_v2
                
                _, detections = detect_solar_panels_v2('temp_orthofoto.png')
                
                self.log(f"✅ Found {len(detections)} potential solar panels")
            else:
                self.log(f"\n{'='*60}")
                self.log("🔍 DETECTING BUILDINGS")
                self.log(f"{'='*60}")
                
                from detect_buildings_fast import detect_buildings_fast
from dotenv import load_dotenv
                
                _, detections = detect_buildings_fast('temp_orthofoto.png')
                
                self.log(f"✅ Found {len(detections)} potential buildings")
            
            # Draw results
            result_img = self.current_image.copy()
            
            threshold = self.confidence_var.get()
            filtered = []
            
            for detection in detections:
                if detection['confidence'] >= threshold:
                    filtered.append(detection)
                    
                    # Get contour and info from detection dict
                    contour = detection['contour']
                    is_rectangular = detection.get('is_rectangular', False)
                    
                    # Draw the actual contour shape
                    if mode == "solar":
                        # Cyan/Yellow for solar panels
                        color = (255, 255, 0) if is_rectangular else (255, 255, 0)
                    else:
                        # Green/Orange for buildings
                        color = (0, 255, 0) if is_rectangular else (0, 165, 255)
                    
                    cv2.drawContours(result_img, [contour], -1, color, 3)
                    
                    # Add label at the top of the bounding box
                    x, y, w, h = detection['bbox']
                    label = f"{detection['confidence']*100:.0f}%"
                    cv2.putText(result_img, label, (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            self.result_image = result_img
            
            # Save with appropriate filename
            if mode == "solar":
                cv2.imwrite('result_solar_panels.png', result_img)
                self.log(f"✅ {len(filtered)} solar panels above {threshold*100:.0f}% confidence")
            else:
                cv2.imwrite('result_all_buildings.png', result_img)
                self.log(f"✅ {len(filtered)} buildings above {threshold*100:.0f}% confidence")
            
            # Update GUI
            self.root.after(0, self.display_results)
            self.root.after(0, lambda: self.update_stats(filtered))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(text="Complete", foreground="green"))
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.notebook.select(1))
            
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(text="Error", foreground="red"))
            self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
    
    def display_results(self):
        """Display results"""
        if self.result_image is None:
            return
        
        img_rgb = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        canvas_width = self.results_canvas.winfo_width()
        canvas_height = self.results_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.results_photo = ImageTk.PhotoImage(img_pil)
        self.results_canvas.delete("all")
        self.results_canvas.create_image(canvas_width//2, canvas_height//2,
                                        image=self.results_photo, anchor=tk.CENTER)
    
    def update_stats(self, detections):
        """Update stats"""
        total = len(detections)
        rectangular = sum(1 for building in detections if building.get('is_rectangular', False))
        irregular = total - rectangular
        
        avg_conf = sum(building['confidence'] for building in detections) / total if total > 0 else 0
        
        stats_text = f"Buildings: {total} | Rectangular: {rectangular} | Irregular: {irregular} | Avg: {avg_conf*100:.1f}%"
        self.stats_label.config(text=stats_text)
    
    def detect_buildings_ndbi(image):
        """
        Detect buildings using NDBI (requires SWIR and NIR bands).
        Returns mask and visualization.
        """
        if image.shape[2] < 5:
            print("❌ NDBI requires SWIR and NIR bands. Your image is RGB only.")
            return None, None
        # Assume band order: [R, G, B, NIR, SWIR]
        nir = image[:, :, 3].astype(np.float32)
        swir = image[:, :, 4].astype(np.float32)
        ndbi = (swir - nir) / (swir + nir + 1e-6)
        mask = (ndbi > 0).astype(np.uint8) * 255
        # Visualization: overlay mask on RGB
        rgb = image[:, :, :3].copy()
        overlay = rgb.copy()
        overlay[mask == 255] = [0, 0, 255]  # Mark buildings in red
        vis = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
        return mask, vis

if __name__ == "__main__":
    root = tk.Tk()
    app = TICDetector(root)
    root.mainloop()

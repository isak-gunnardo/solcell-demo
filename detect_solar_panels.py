import cv2
import numpy as np
import pickle
import os

def detect_solar_panels(image_path, min_area=200, max_area=50000):
    """
    Detect solar panels using computer vision + ML (if trained)
    Solar panels have distinctive characteristics:
    - Dark blue/black color (low saturation, low value)
    - Rectangular shapes with parallel edges
    - Grid-like texture
    - High contrast with surrounding roof
    """
    print("☀️ SOLAR PANEL DETECTOR")
    print("=" * 70)
    
    # Try to load trained model
    use_ml = False
    clf = None
    scaler = None
    try:
        if os.path.exists('solar_panel_classifier.pkl'):
            with open('solar_panel_classifier.pkl', 'rb') as f:
                clf, scaler = pickle.load(f)
            print("🤖 Using TRAINED ML MODEL")
            use_ml = True
        else:
            print("🎨 Using COMPUTER VISION only (no trained model)")
    except:
        print("🎨 Using COMPUTER VISION only (model load failed)")
    
    print("=" * 70)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None, []
    
    original = img.copy()
    height, width = img.shape[:2]
    print(f"📐 Image: {width}x{height}")
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # === STEP 1: COLOR-BASED DETECTION ===
    print("🎨 Step 1: Detecting dark blue/black surfaces (solar panels)...")
    
    # Solar panels are typically:
    # - Dark (low V value in HSV)
    # - Low saturation (appears grey/black)
    # - Blue hue (if visible)
    
    # Adjusted for your labeled data (V around 85-95)
    # Method 1: Moderately dark regions  
    dark_mask = (v < 110).astype(np.uint8) * 255
    
    # Method 2: Blue-ish regions (based on your labels: V < 110, S > 50)
    blue_mask = ((h >= 90) & (h <= 130) & (v < 110) & (s > 50)).astype(np.uint8) * 255
    
    # Method 3: Dark with moderate saturation
    moderate_dark_mask = ((s > 60) & (v < 100)).astype(np.uint8) * 255
    
    # Combine masks
    solar_mask = cv2.bitwise_or(dark_mask, blue_mask)
    solar_mask = cv2.bitwise_or(solar_mask, moderate_dark_mask)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # === STEP 2: EDGE DETECTION ===
    print("📐 Step 2: Detecting rectangular edges...")
    
    # Enhanced edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby segments
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel_edge, iterations=1)
    
    # Combine with color mask
    combined = cv2.bitwise_and(solar_mask, edges)
    
    # === STEP 3: FIND CONTOURS ===
    print("🔍 Step 3: Finding rectangular shapes...")
    
    contours, _ = cv2.findContours(solar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} potential regions")
    
    # === STEP 4: FILTER BY SHAPE AND SIZE ===
    print("🔧 Step 4: Filtering for solar panel characteristics...")
    
    solar_panels = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Size filtering
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small regions
        if w < 5 or h < 5:
            continue
        
        # Aspect ratio (solar panels are usually rectangular)
        aspect_ratio = max(w, h) / min(w, h)
        
        # Solar panels typically have aspect ratio between 1:1 and 4:1
        if aspect_ratio > 6:
            continue
        
        # Calculate rectangularity (how close to rectangle)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Solar panels should be fairly rectangular (extent > 0.6)
        if extent < 0.5:
            continue
        
        # Approximate polygon to check if rectangular
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # Check for 4 corners (rectangular)
        is_rectangular = len(approx) == 4
        
        # Calculate confidence
        if use_ml:
            # === USE ML MODEL ===
            # Extract features
            roi_gray = gray[y:y+h, x:x+w]
            roi_hsv = hsv[y:y+h, x:x+w]
            roi_lab = lab[y:y+h, x:x+w]
            
            if roi_gray.size == 0:
                continue
            
            h_ch, s_ch, v_ch = cv2.split(roi_hsv)
            l_ch, a_ch, b_ch = cv2.split(roi_lab)
            
            features = [
                np.mean(h_ch), np.std(h_ch),
                np.mean(s_ch), np.std(s_ch),
                np.mean(v_ch), np.std(v_ch),
                np.mean(l_ch), np.std(l_ch),
                np.mean(a_ch), np.std(a_ch),
                np.mean(b_ch), np.std(b_ch),
                np.percentile(v_ch, 25),
                np.percentile(v_ch, 50),
                np.percentile(v_ch, 75),
                np.mean(edges[y:y+h, x:x+w]),
                np.std(edges[y:y+h, x:x+w]),
                np.sum(edges[y:y+h, x:x+w] > 0) / (w * h),
                w, h, area, aspect_ratio,
                np.sum((h_ch >= 90) & (h_ch <= 130)) / h_ch.size,
                np.sum(s_ch < 50) / s_ch.size,
            ]
            
            features_scaled = scaler.transform([features])
            ml_confidence = clf.predict_proba(features_scaled)[0][1]  # Probability of solar panel
            
            # Lower threshold since your panels aren't very dark
            if ml_confidence < 0.50:
                continue
            
            confidence = ml_confidence
            avg_darkness = 1.0 - (np.mean(v_ch) / 255.0)
        else:
            # === USE HEURISTICS ===
            confidence = 0.0
            
            # Factor 1: Color darkness (darker = more likely solar panel)
            roi_v = v[y:y+h, x:x+w]
            avg_darkness = 1.0 - (np.mean(roi_v) / 255.0)
            confidence += avg_darkness * 0.4
            
            # Factor 2: Rectangularity
            confidence += extent * 0.3
            
            # Factor 3: Shape (4 corners bonus)
            if is_rectangular:
                confidence += 0.2
            else:
                confidence += 0.1
            
            # Factor 4: Edge strength
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / (w * h)
            confidence += min(edge_density * 2, 0.1)
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
        
        # Store panel info
        panel = {
            'contour': contour,
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'is_rectangular': is_rectangular,
            'confidence': confidence,
            'avg_darkness': avg_darkness
        }
        
        solar_panels.append(panel)
    
    # === STEP 5: REMOVE OVERLAPPING DETECTIONS ===
    print("🔄 Step 5: Removing overlaps...")
    
    def boxes_overlap(box1, box2, threshold=0.5):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        smaller = min(area1, area2)
        
        return (intersection / smaller) > threshold
    
    # Sort by confidence (keep higher confidence detections)
    solar_panels.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for panel in solar_panels:
        is_dup = False
        for kept in filtered:
            if boxes_overlap(panel['bbox'], kept['bbox']):
                is_dup = True
                break
        if not is_dup:
            filtered.append(panel)
    
    solar_panels = filtered
    print(f"   Final count: {len(solar_panels)} solar panels")
    
    # === STEP 6: DRAW RESULTS ===
    print("🎨 Step 6: Drawing results...")
    
    result = original.copy()
    
    for idx, panel in enumerate(solar_panels):
        contour = panel['contour']
        x, y, w, h = panel['bbox']
        confidence = panel['confidence']
        is_rectangular = panel.get('is_rectangular', False)
        
        # Draw contour - CYAN for rectangular, YELLOW for irregular
        color = (255, 255, 0) if is_rectangular else (0, 255, 255)
        cv2.drawContours(result, [contour], 0, color, 3)
        
        # Draw bounding box (lighter color)
        box_color = (200, 200, 100) if is_rectangular else (100, 200, 200)
        cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 1)
        
        # Label
        label = f"☀️#{idx+1} ({confidence:.0%})"
        cv2.putText(result, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save result
    output_path = 'result_solar_panels.png'
    cv2.imwrite(output_path, result)
    
    # === STATISTICS ===
    print("\n" + "=" * 70)
    print(f"☀️ FINAL RESULTS: {len(solar_panels)} solar panels detected")
    print("=" * 70)
    
    if solar_panels:
        confidences = [p['confidence'] for p in solar_panels]
        areas = [p['area'] for p in solar_panels]
        darkness = [p['avg_darkness'] for p in solar_panels]
        
        rectangular_count = sum(1 for p in solar_panels if p.get('is_rectangular', False))
        
        print(f"\n📊 STATISTICS:")
        print(f"Total panels: {len(solar_panels)}")
        print(f"Rectangular: {rectangular_count} ({rectangular_count/len(solar_panels)*100:.0f}%)")
        print(f"Confidence range: {min(confidences):.0%} - {max(confidences):.0%}")
        print(f"Average confidence: {np.mean(confidences):.0%}")
        print(f"Size range: {min(areas):.0f} - {max(areas):.0f} pixels")
        print(f"Average size: {np.mean(areas):.0f} pixels")
        print(f"Average darkness: {np.mean(darkness):.0%}")
        
        print(f"\n☀️ DETECTED SOLAR PANELS:")
        print("-" * 80)
        print(" # | Position  | Size    | Area   | Conf | Dark | Rect | Shape")
        print("-" * 80)
        for idx, p in enumerate(solar_panels):
            x, y, w, h = p['bbox']
            shape_type = "Rectangular" if p.get('is_rectangular', False) else "Irregular"
            print(f"{idx+1:2d} | ({x:4d},{y:4d}) | {w:3d}×{h:3d} | {p['area']:6.0f} | "
                  f"{p['confidence']:4.0%} | {p['avg_darkness']:3.0%} | "
                  f"{p['aspect_ratio']:4.1f} | {shape_type}")
    
    print(f"\n✅ Saved: '{output_path}'")
    
    return result, solar_panels

if __name__ == "__main__":
    try:
        result, panels = detect_solar_panels("temp_orthofoto.png")
        if result is not None:
            print("\n✅ Detection complete!")
            cv2.imshow("Solar Panel Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

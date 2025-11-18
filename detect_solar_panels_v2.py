import cv2
import numpy as np
import pickle
import os

def detect_solar_panels_v2(image_path, min_area=200, max_area=50000):
    """
    Detect solar panels by looking for:
    1. Rectangular dark regions (panel core)
    2. Lighter frames around them
    3. Grid-like texture
    """
    print("☀️ SOLAR PANEL DETECTOR V2 (Frame-Based)")
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
            print("🎨 Using COMPUTER VISION only")
    except:
        print("🎨 Using COMPUTER VISION only")
    
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
    l, a, b_ch = cv2.split(lab)
    
    # === STEP 1: DETECT FRAMES (Edge-based approach) ===
    print("🔲 Step 1: Detecting rectangular frames...")
    
    # Enhanced edge detection for frames
    edges = cv2.Canny(gray, 30, 100)
    
    # Find lines (potential frame edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
    
    # Dilate edges to emphasize frames
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_rect, iterations=1)
    
    # === STEP 2: FIND ENCLOSED RECTANGLES ===
    print("📦 Step 2: Finding enclosed rectangular regions...")
    
    # Find contours that might be frames
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} potential frame regions")
    
    # === STEP 3: ANALYZE EACH REGION ===
    print("🔍 Step 3: Analyzing regions for solar panel characteristics...")
    
    solar_panels = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Size filtering
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very small or edge regions
        if w < 10 or h < 10 or x < 5 or y < 5 or x+w >= width-5 or y+h >= height-5:
            continue
        
        # Aspect ratio (solar panels are rectangular)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        # Car filtering: cars are small and square
        if area < 300 and 0.7 < aspect_ratio < 1.3:
            continue
        
        # Solar panels typically have aspect ratio between 1:1 and 4:1
        if aspect_ratio > 5:
            continue
        
        # Calculate rectangularity
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        if extent < 0.4:
            continue
        
        # Approximate polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        is_rectangular = len(approx) == 4
        
        # === KEY FEATURE: DETECT FRAME ===
        # Check if region has darker center with lighter border
        roi_gray = gray[y:y+h, x:x+w]
        roi_v = v[y:y+h, x:x+w]
        
        if roi_gray.size == 0:
            continue
        
        # Divide region into center and border
        border_size = max(2, min(w, h) // 8)  # Border is ~12% of size
        
        if w > border_size*4 and h > border_size*4:
            # Inner core (should be darker)
            inner_v = roi_v[border_size:-border_size, border_size:-border_size]
            
            # Outer border (might be lighter)
            top_border = roi_v[:border_size, :]
            bottom_border = roi_v[-border_size:, :]
            left_border = roi_v[:, :border_size]
            right_border = roi_v[:, -border_size:]
            
            if inner_v.size > 0:
                inner_mean = np.mean(inner_v)
                border_mean = np.mean([
                    np.mean(top_border),
                    np.mean(bottom_border),
                    np.mean(left_border),
                    np.mean(right_border)
                ])
                
                # Solar panel: dark center, lighter frame
                has_frame = border_mean > inner_mean + 10  # Frame at least 10 brightness units lighter
            else:
                inner_mean = np.mean(roi_v)
                border_mean = inner_mean
                has_frame = False
        else:
            inner_mean = np.mean(roi_v)
            border_mean = inner_mean
            has_frame = False
        
        # Overall darkness
        avg_v = np.mean(roi_v)
        avg_darkness = 1.0 - (avg_v / 255.0)

        # Color filtering based on real solar panel pixel
        roi_h = hsv[y:y+h, x:x+w, 0]
        roi_s = hsv[y:y+h, x:x+w, 1]
        roi_lab_l = lab[y:y+h, x:x+w, 0]
        mean_h = np.mean(roi_h)
        mean_s = np.mean(roi_s)
        mean_lab_l = np.mean(roi_lab_l)

        # Only keep regions matching real panel color
        # Expanded thresholds based on user pixel samples
        # Stricter thresholds to avoid buildings
        if not (mean_h >= 90 and mean_h <= 110 and mean_s > 40 and avg_v < 100 and mean_lab_l < 90):
            continue
        
        # === DETECT GRID PATTERN (solar cells) ===
        # Solar panels have internal grid structure
        roi_edges = edges[y:y+h, x:x+w]
        edge_density = np.sum(roi_edges > 0) / (w * h)
        
        # Grid detection: look for regular internal edges
        has_grid = edge_density > 0.05 and edge_density < 0.3
        
        # === CALCULATE CONFIDENCE ===
        if use_ml:
            # Extract features for ML
            roi_hsv = hsv[y:y+h, x:x+w]
            roi_lab = lab[y:y+h, x:x+w]
            
            h_ch, s_ch, v_ch = cv2.split(roi_hsv)
            l_ch, a_ch, b_lab = cv2.split(roi_lab)
            
            features = [
                np.mean(h_ch), np.std(h_ch),
                np.mean(s_ch), np.std(s_ch),
                np.mean(v_ch), np.std(v_ch),
                np.mean(l_ch), np.std(l_ch),
                np.mean(a_ch), np.std(a_ch),
                np.mean(b_lab), np.std(b_lab),
                np.percentile(v_ch, 25),
                np.percentile(v_ch, 50),
                np.percentile(v_ch, 75),
                np.mean(edges[y:y+h, x:x+w]),
                np.std(edges[y:y+h, x:x+w]),
                edge_density,
                w, h, area, aspect_ratio,
                np.sum((h_ch >= 90) & (h_ch <= 130)) / h_ch.size,
                np.sum(s_ch < 50) / s_ch.size,
            ]
            
            features_scaled = scaler.transform([features])
            ml_confidence = clf.predict_proba(features_scaled)[0][1]
            
            # Boost confidence if has frame or grid
            confidence = ml_confidence
            if has_frame:
                confidence = min(1.0, confidence + 0.15)
            if has_grid:
                confidence = min(1.0, confidence + 0.10)
            
            # Still require minimum ML confidence
            if ml_confidence < 0.40:
                continue
        else:
            # Heuristic confidence
            confidence = 0.0
            
            # Factor 1: Darkness (30%)
            confidence += avg_darkness * 0.3
            
            # Factor 2: Rectangularity (20%)
            confidence += extent * 0.2
            
            # Factor 3: Has visible frame (25% bonus!)
            if has_frame:
                confidence += 0.25
            
            # Factor 4: Has grid pattern (15% bonus!)
            if has_grid:
                confidence += 0.15
            
            # Factor 5: Shape (10%)
            if is_rectangular:
                confidence += 0.10
            
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
            'avg_darkness': avg_darkness,
            'has_frame': has_frame,
            'has_grid': has_grid,
            'inner_brightness': inner_mean,
            'frame_brightness': border_mean
        }
        
        solar_panels.append(panel)
    
    # === STEP 4: REMOVE OVERLAPS ===
    print("🔄 Step 4: Removing overlapping detections...")
    
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
    
    # Sort by confidence
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
    
    # === STEP 5: DRAW RESULTS ===
    print("🎨 Step 5: Drawing results...")
    
    result = original.copy()
    
    for idx, panel in enumerate(solar_panels):
        contour = panel['contour']
        x, y, w, h = panel['bbox']
        confidence = panel['confidence']
        has_frame = panel.get('has_frame', False)
        has_grid = panel.get('has_grid', False)
        
        # Color based on features
        if has_frame and has_grid:
            color = (0, 255, 0)  # Green: has both frame and grid
        elif has_frame:
            color = (0, 255, 255)  # Yellow: has frame
        elif has_grid:
            color = (255, 128, 0)  # Orange: has grid
        else:
            color = (255, 0, 255)  # Magenta: basic detection
        
        # Draw contour
        cv2.drawContours(result, [contour], 0, color, 2)
        
        # Label
        label = f"#{idx+1} {confidence:.0%}"
        if has_frame:
            label += " F"
        if has_grid:
            label += " G"
        
        cv2.putText(result, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save
    output_path = 'result_solar_panels.png'
    cv2.imwrite(output_path, result)
    
    # === STATISTICS ===
    print("\n" + "=" * 70)
    print(f"☀️ FINAL RESULTS: {len(solar_panels)} solar panels detected")
    print("=" * 70)
    
    if solar_panels:
        with_frame = sum(1 for p in solar_panels if p.get('has_frame', False))
        with_grid = sum(1 for p in solar_panels if p.get('has_grid', False))
        
        print(f"\n📊 STATISTICS:")
        print(f"Total panels: {len(solar_panels)}")
        print(f"With visible frame: {with_frame} ({with_frame/len(solar_panels)*100:.0f}%)")
        print(f"With grid pattern: {with_grid} ({with_grid/len(solar_panels)*100:.0f}%)")
        
        confidences = [p['confidence'] for p in solar_panels]
        print(f"Confidence range: {min(confidences):.0%} - {max(confidences):.0%}")
        print(f"Average confidence: {np.mean(confidences):.0%}")
        
        print(f"\n☀️ DETECTED SOLAR PANELS:")
        print("-" * 80)
        print(" # | Position  | Size    | Conf | Frame | Grid | Inner-V | Border-V")
        print("-" * 80)
        for idx, p in enumerate(solar_panels[:20]):  # Show first 20
            x, y, w, h = p['bbox']
            frame_marker = "✓" if p.get('has_frame') else "✗"
            grid_marker = "✓" if p.get('has_grid') else "✗"
            inner_v = p.get('inner_brightness', 0)
            border_v = p.get('frame_brightness', 0)
            print(f"{idx+1:2d} | ({x:4d},{y:4d}) | {w:3d}×{h:3d} | {p['confidence']:4.0%} | "
                  f"  {frame_marker}   |  {grid_marker}  | {inner_v:7.1f} | {border_v:8.1f}")
    
    print(f"\n✅ Saved: '{output_path}'")
    print("\n💡 Color Legend:")
    print("   🟢 Green = Has frame AND grid")
    print("   🟡 Yellow = Has frame")
    print("   🟠 Orange = Has grid")
    print("   🟣 Magenta = Basic detection")
    
    return result, solar_panels

if __name__ == "__main__":
    try:
        result, panels = detect_solar_panels_v2("temp_orthofoto.png")
        if result is not None:
            print("\n✅ Detection complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

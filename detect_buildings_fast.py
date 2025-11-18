import cv2
import numpy as np
import pickle

def detect_buildings_fast(image_path, model_path='building_classifier.pkl'):
    """
    Fast building detection using trained model
    Strategy: Find candidate regions first, then classify them
    """
    print("🤖 FAST TRAINED MODEL DETECTOR")
    print("=" * 70)
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            clf, scaler = pickle.load(f)
        print("✅ Loaded trained model")
    except FileNotFoundError:
        print("❌ Model not found! Run 'python label_buildings.py' first.")
        return
    
    # Load image
    img = cv2.imread(image_path)
    original = img.copy()
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    print(f"📐 Image: {width}x{height}")
    
    # === STEP 1: FIND CANDIDATE REGIONS ===
    print("🔍 Step 1: Finding candidate regions...")
    
    # Remove obvious non-buildings (vegetation)
    h, s, v = cv2.split(hsv)
    vegetation = ((h >= 30) & (h <= 90) & (s >= 15)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    vegetation = cv2.dilate(vegetation, kernel, iterations=2)
    
    # Detect roads (grey, low saturation, elongated) - LESS AGGRESSIVE
    print("🛣️  Step 1b: Detecting and excluding roads...")
    roads_mask = ((s < 15) & (v > 90) & (v < 160)).astype(np.uint8) * 255
    
    # Find elongated structures using morphological operations - SMALLER KERNELS
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))  # Vertical - longer
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))  # Horizontal - longer
    vertical_roads = cv2.morphologyEx(roads_mask, cv2.MORPH_OPEN, kernel_v)
    horizontal_roads = cv2.morphologyEx(roads_mask, cv2.MORPH_OPEN, kernel_h)
    roads = cv2.bitwise_or(vertical_roads, horizontal_roads)
    
    # Expand roads less aggressively
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    roads = cv2.dilate(roads, kernel_expand, iterations=1)
    
    # Find bright AND dark areas (to catch grey/black roofs) - MORE INCLUSIVE
    _, thresh_bright = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    _, thresh_dark = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_or(thresh_bright, thresh_dark)
    
    # Add edge detection to find rectangular boundaries
    print("📐 Step 1c: Detecting rectangular edges...")
    edges = cv2.Canny(gray, 50, 150)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel_rect, iterations=1)
    
    # Combine threshold and edges
    thresh = cv2.bitwise_or(thresh, edges)
    
    # Remove both vegetation and roads from candidates
    exclusion = cv2.bitwise_or(vegetation, roads)
    candidates = cv2.bitwise_and(thresh, cv2.bitwise_not(exclusion))
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(contours)} candidate regions")
    
    # === STEP 1c: DETECT RECTANGULAR SHAPES (ROOFS) ===
    print("🏠 Step 1c: Detecting rectangular roof shapes...")
    rectangular_indices = set()
    
    for idx, contour in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4-6 vertices)
        # Allow 5-6 vertices for slightly irregular rectangles
        if len(approx) >= 4 and len(approx) <= 6:
            # Calculate how rectangular it is
            area = cv2.contourArea(contour)
            if area > 150:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0
                
                # Good rectangles have extent > 0.45 (fill good portion of bounding box)
                if extent > 0.45:
                    rectangular_indices.add(idx)
    
    print(f"   Found {len(rectangular_indices)} rectangular shapes (potential roofs)")
    
    # Prioritize rectangular contours
    all_contours = []
    for idx in rectangular_indices:
        all_contours.append((contours[idx], True))  # (contour, is_rectangular)
    for idx, contour in enumerate(contours):
        if idx not in rectangular_indices:
            all_contours.append((contour, False))
    
    # === STEP 2: CLASSIFY EACH CANDIDATE ===
    print("🤖 Step 2: Classifying candidates with trained model...")
    
    buildings = []
    
    for contour_data in all_contours:
        contour, is_rectangular = contour_data
        area = cv2.contourArea(contour)
        
        # Size filter - VERY LENIENT
        if area < 100 or area > width * height * 0.35:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum size - VERY LENIENT
        if w < 10 or h < 10:
            continue
        
        # Aspect ratio - roads are very elongated, buildings are not
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # CRITICAL ROAD FILTER: Check if this overlaps significantly with road mask
        roi_mask = np.zeros_like(gray)
        cv2.drawContours(roi_mask, [contour], 0, 255, -1)
        road_overlap = cv2.bitwise_and(roi_mask, roads)
        overlap_ratio = np.sum(road_overlap > 0) / np.sum(roi_mask > 0) if np.sum(roi_mask > 0) > 0 else 0
        
        # If it's highly elongated AND overlaps with roads -> it's a road
        is_elongated = aspect_ratio < 0.12 or aspect_ratio > 8.0
        if is_elongated and overlap_ratio > 0.5:
            continue
        
        # If it overlaps heavily with roads (even if not elongated) -> it's a road
        if overlap_ratio > 0.75:
            continue
        
        # Standard aspect ratio filter (allow wide range for buildings)
        if aspect_ratio < 0.08 or aspect_ratio > 12.0:
            continue
        
        # Calculate extent - VERY RELAXED
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < 0.12:  # Very lenient
            continue
        
        # Calculate compactness - VERY RELAXED
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Very low compactness + high road overlap = road
        if compactness < 0.06 and overlap_ratio > 0.5:
            continue
        
        # Reasonable compactness for buildings - VERY RELAXED
        if compactness < 0.06:
            continue
        
        # Extract features for this region
        roi_gray = gray[y:y+h, x:x+w]
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_lab = lab[y:y+h, x:x+w]
        
        features = []
        
        # Gray statistics
        features.append(np.mean(roi_gray))
        features.append(np.std(roi_gray))
        features.append(np.median(roi_gray))
        features.append(np.percentile(roi_gray, 25))
        features.append(np.percentile(roi_gray, 75))
        
        # HSV statistics
        h_roi, s_roi, v_roi = cv2.split(roi_hsv)
        features.append(np.mean(h_roi))
        features.append(np.std(h_roi))
        features.append(np.mean(s_roi))
        features.append(np.std(s_roi))
        features.append(np.mean(v_roi))
        features.append(np.std(v_roi))
        
        # LAB statistics
        l, a, b = cv2.split(roi_lab)
        features.append(np.mean(l))
        features.append(np.std(l))
        features.append(np.mean(a))
        features.append(np.std(a))
        features.append(np.mean(b))
        features.append(np.std(b))
        
        # Texture
        edges = cv2.Canny(roi_gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)
        
        # Color variance
        color_var = np.var(roi_hsv.reshape(-1, 3), axis=0)
        features.extend(color_var)
        
        # Predict
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = clf.predict(features_scaled)[0]
        confidence = clf.predict_proba(features_scaled)[0][1]
        
        # Lower threshold for rectangular shapes (they're more likely buildings)
        confidence_threshold = 0.35 if is_rectangular else 0.45
        
        if prediction == 1 and confidence > confidence_threshold:
            buildings.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'confidence': confidence,
                'aspect_ratio': aspect_ratio,
                'is_rectangular': is_rectangular
            })
    
    print(f"   Model classified {len(buildings)} as buildings")
    
    # === STEP 3: REMOVE OVERLAPS ===
    print("🔄 Step 3: Removing overlapping detections...")
    
    # Sort by confidence
    buildings.sort(key=lambda x: x['confidence'], reverse=True)
    
    def boxes_overlap(box1, box2, threshold=0.4):
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
    
    filtered = []
    for building in buildings:
        is_dup = False
        for kept in filtered:
            if boxes_overlap(building['bbox'], kept['bbox']):
                is_dup = True
                break
        if not is_dup:
            filtered.append(building)
    
    buildings = filtered
    print(f"   Final count: {len(buildings)} buildings")
    
    # === STEP 4: DRAW RESULTS ===
    print("🎨 Step 4: Drawing results...")
    
    result = original.copy()
    
    for idx, building in enumerate(buildings):
        contour = building['contour']
        x, y, w, h = building['bbox']
        confidence = building['confidence']
        is_rectangular = building.get('is_rectangular', False)
        
        # Draw contour - CYAN for rectangular roofs, GREEN for others
        color = (255, 255, 0) if is_rectangular else (0, 255, 0)
        cv2.drawContours(result, [contour], 0, color, 3)
        
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Label with rectangle indicator
        rect_marker = "□" if is_rectangular else ""
        label = f"#{idx+1}{rect_marker} ({confidence:.0%})"
        cv2.putText(result, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save
    output_path = 'result_all_buildings.png'
    cv2.imwrite(output_path, result)
    
    # === STATISTICS ===
    print("\n" + "=" * 70)
    print(f"🏠 FINAL RESULTS: {len(buildings)} buildings detected")
    print("=" * 70)
    
    if buildings:
        confidences = [b['confidence'] for b in buildings]
        areas = [b['area'] for b in buildings]
        
        rectangular_count = sum(1 for b in buildings if b.get('is_rectangular', False))
        
        print(f"\n📊 STATISTICS:")
        print(f"Total buildings: {len(buildings)}")
        print(f"Rectangular roofs: {rectangular_count} ({rectangular_count/len(buildings)*100:.0f}%)")
        print(f"Confidence range: {min(confidences):.0%} - {max(confidences):.0%}")
        print(f"Average confidence: {np.mean(confidences):.0%}")
        print(f"Size range: {min(areas):.0f} - {max(areas):.0f} pixels")
        print(f"Average size: {np.mean(areas):.0f} pixels")
        
        print(f"\n🏗️ ALL DETECTED BUILDINGS:")
        print("-" * 76)
        print(" # | Position  | Size    | Area   | Conf | Ratio | Shape")
        print("-" * 76)
        for idx, b in enumerate(buildings):
            x, y, w, h = b['bbox']
            shape_type = "Rectangular" if b.get('is_rectangular', False) else "Irregular"
            print(f"{idx+1:2d} | ({x:4d},{y:4d}) | {w:3d}×{h:3d} | {b['area']:6.0f} | "
                  f"{b['confidence']:4.0%} | {b['aspect_ratio']:.2f} | {shape_type}")
    
    print(f"\n✅ Saved: '{output_path}'")
    
    target = 23
    accuracy = len(buildings) / target * 100
    print(f"\n📊 DETECTION RATE: {len(buildings)}/{target} buildings ({accuracy:.1f}%)")
    
    if len(buildings) >= target:
        print(f"   🎯 Excellent!")
        if len(buildings) > target:
            print(f"   ⚠️  {len(buildings) - target} possible false positives")
    else:
        print(f"   📉 Missing {target - len(buildings)} buildings")
    
    print("\n💡 To improve results:")
    print("   - Run 'python label_buildings.py' to add more training examples")
    print("   - Label buildings the model is missing")
    print("   - Label false positives as non-buildings")
    
    return result, buildings

if __name__ == "__main__":
    try:
        result, buildings = detect_buildings_fast("ortofoto.png")
        print("\n✅ Detection complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Global variables
img = None
original = None
buildings = []
non_buildings = []
current_label = 1
drawing = False
ix, iy = -1, -1

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img, original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        if x2 - x1 > 10 and y2 - y1 > 10:
            if current_label == 1:
                buildings.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"B{len(buildings)}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"✅ Added building #{len(buildings)}")
            else:
                non_buildings.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"N{len(non_buildings)}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"✅ Added non-building #{len(non_buildings)}")
        
        cv2.imshow('Label Buildings', img)

def extract_features(img_gray, img_hsv, img_lab, x1, y1, x2, y2):
    """Extract features from a region"""
    roi_gray = img_gray[y1:y2, x1:x2]
    roi_hsv = img_hsv[y1:y2, x1:x2]
    roi_lab = img_lab[y1:y2, x1:x2]
    
    if roi_gray.size == 0:
        return None
    
    features = []
    
    # Gray statistics
    features.append(np.mean(roi_gray))
    features.append(np.std(roi_gray))
    features.append(np.median(roi_gray))
    features.append(np.percentile(roi_gray, 25))
    features.append(np.percentile(roi_gray, 75))
    
    # HSV statistics
    h, s, v = cv2.split(roi_hsv)
    features.append(np.mean(h))
    features.append(np.std(h))
    features.append(np.mean(s))
    features.append(np.std(s))
    features.append(np.mean(v))
    features.append(np.std(v))
    
    # LAB statistics
    l, a, b = cv2.split(roi_lab)
    features.append(np.mean(l))
    features.append(np.std(l))
    features.append(np.mean(a))
    features.append(np.std(a))
    features.append(np.mean(b))
    features.append(np.std(b))
    
    # Texture features (edges)
    edges = cv2.Canny(roi_gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    
    # Color variance
    if len(roi_hsv.shape) == 3:
        color_var = np.var(roi_hsv.reshape(-1, roi_hsv.shape[2]), axis=0)
        features.extend(color_var)
    
    return np.array(features)

def main():
    global img, original, current_label
    
    print("=" * 70)
    print("🏗️ IMPROVED BUILDING LABELING TOOL")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Label buildings with BLACK/GREY roofs!")
    print("   Your model is missing these because training data was biased")
    print("   towards red/orange roofs.\n")
    print("📋 Instructions:")
    print("   1. Draw rectangles around BLACK/GREY roofed buildings (green)")
    print("   2. Also label a few RED/ORANGE roofed buildings")
    print("   3. Press 'n' to switch to NON-BUILDINGS mode (red)")
    print("   4. Label trees, grass, roads, fences as non-buildings")
    print("   5. Press 'b' to switch back to BUILDINGS mode")
    print("   6. Press 'u' to UNDO last label")
    print("   7. Press 's' to SAVE training data")
    print("   8. Press 't' to TRAIN the model")
    print("   9. Press 'q' to QUIT")
    print("\n🎯 Goal: Label 10+ black/grey buildings, 5+ red buildings, 10+ non-buildings")
    print("=" * 70)
    
    # Load existing model to show what it learned
    try:
        with open('building_classifier.pkl', 'rb') as f:
            clf, scaler = pickle.load(f)
        print("\n✅ Loaded existing model - you're ADDING to previous training")
        
        # Try to load previous labels
        try:
            with open('training_labels.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                buildings.extend(saved_data['buildings'])
                non_buildings.extend(saved_data['non_buildings'])
            print(f"✅ Loaded previous labels: {len(buildings)} buildings, {len(non_buildings)} non-buildings")
        except:
            print("📝 Starting fresh labels (previous labels not found)")
    except:
        print("\n📝 No existing model - starting from scratch")
    
    # Load image (try temp_orthofoto.png first, then ortofoto.png)
    original = cv2.imread('temp_orthofoto.png')
    if original is None:
        original = cv2.imread('ortofoto.png')
    if original is None:
        print("\n❌ ERROR: No image found!")
        print("   Please run tic_simple.py first to download an orthofoto image.")
        print("   OR place your ortofoto.png in this directory.")
        return
    img = original.copy()
    
    # Redraw existing labels
    for idx, (x1, y1, x2, y2) in enumerate(buildings):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"B{idx+1}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for idx, (x1, y1, x2, y2) in enumerate(non_buildings):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"N{idx+1}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.namedWindow('Label Buildings')
    cv2.setMouseCallback('Label Buildings', mouse_callback)
    cv2.imshow('Label Buildings', img)
    
    mode_text = "BUILDING MODE (Green)"
    
    while True:
        display = img.copy()
        cv2.putText(display, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Buildings: {len(buildings)} | Non-buildings: {len(non_buildings)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Label Buildings', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('n'):
            current_label = 0
            mode_text = "NON-BUILDING MODE (Red)"
            print("\n🔴 Switched to NON-BUILDING mode")
            
        elif key == ord('b'):
            current_label = 1
            mode_text = "BUILDING MODE (Green)"
            print("\n🟢 Switched to BUILDING mode")
            
        elif key == ord('u'):
            if current_label == 1 and buildings:
                buildings.pop()
                print(f"↩️  Removed last building. Total: {len(buildings)}")
            elif current_label == 0 and non_buildings:
                non_buildings.pop()
                print(f"↩️  Removed last non-building. Total: {len(non_buildings)}")
            
            # Redraw
            img = original.copy()
            for idx, (x1, y1, x2, y2) in enumerate(buildings):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"B{idx+1}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for idx, (x1, y1, x2, y2) in enumerate(non_buildings):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"N{idx+1}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        elif key == ord('s'):
            # Save labels
            with open('training_labels.pkl', 'wb') as f:
                pickle.dump({
                    'buildings': buildings,
                    'non_buildings': non_buildings
                }, f)
            print(f"\n💾 Saved {len(buildings)} buildings and {len(non_buildings)} non-buildings")
            
        elif key == ord('t'):
            if len(buildings) < 5 or len(non_buildings) < 5:
                print("\n⚠️  Need at least 5 buildings and 5 non-buildings!")
                continue
            
            print("\n" + "=" * 70)
            print("🤖 TRAINING IMPROVED MODEL...")
            print("=" * 70)
            
            cv2.destroyAllWindows()
            
            # Extract features
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            
            X = []
            y = []
            
            print(f"📊 Extracting features from {len(buildings)} buildings...")
            for x1, y1, x2, y2 in buildings:
                features = extract_features(gray, hsv, lab, x1, y1, x2, y2)
                if features is not None:
                    X.append(features)
                    y.append(1)
            
            print(f"📊 Extracting features from {len(non_buildings)} non-buildings...")
            for x1, y1, x2, y2 in non_buildings:
                features = extract_features(gray, hsv, lab, x1, y1, x2, y2)
                if features is not None:
                    X.append(features)
                    y.append(0)
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"\n✅ Training data prepared:")
            print(f"   - {np.sum(y==1)} building samples")
            print(f"   - {np.sum(y==0)} non-building samples")
            print(f"   - {X.shape[1]} features per sample")
            
            # Train improved Random Forest with more trees
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            clf = RandomForestClassifier(
                n_estimators=200,  # More trees
                max_depth=15,      # Deeper trees
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
            clf.fit(X_scaled, y)
            
            train_acc = clf.score(X_scaled, y) * 100
            print(f"\n🎯 Training accuracy: {train_acc:.1f}%")
            
            # Save model
            with open('building_classifier.pkl', 'wb') as f:
                pickle.dump((clf, scaler), f)
            
            print("✅ Improved model saved!")
            
            # Save labels too
            with open('training_labels.pkl', 'wb') as f:
                pickle.dump({
                    'buildings': buildings,
                    'non_buildings': non_buildings
                }, f)
            
            print("\n" + "=" * 70)
            print("🚀 MODEL TRAINING COMPLETE!")
            print("=" * 70)
            print("\n📋 Next steps:")
            print("   1. Run: python detect_buildings_fast.py")
            print("   2. Check if black/grey buildings are now detected")
            print("   3. If still missing some, label more and train again")
            print("=" * 70)
            
            return
    
    cv2.destroyAllWindows()
    print("\n❌ Training cancelled")

if __name__ == "__main__":
    main()

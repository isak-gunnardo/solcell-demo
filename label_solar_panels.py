import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Global variables
img = None
original = None
solar_panels = []
non_panels = []
current_label = 1  # 1=solar panel, 0=non-panel
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
                solar_panels.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for solar
                cv2.putText(img, f"S{len(solar_panels)}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                print(f"☀️ Added solar panel #{len(solar_panels)}")
            else:
                non_panels.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for non-panel
                cv2.putText(img, f"N{len(non_panels)}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"❌ Added non-panel #{len(non_panels)}")
        
        cv2.imshow('Label Solar Panels', img)

def extract_features(img_gray, img_hsv, img_lab, x1, y1, x2, y2):
    """Extract features from a region"""
    roi_gray = img_gray[y1:y2, x1:x2]
    roi_hsv = img_hsv[y1:y2, x1:x2]
    roi_lab = img_lab[y1:y2, x1:x2]
    
    if roi_gray.size == 0:
        return None
    
    h, s, v = cv2.split(roi_hsv)
    l, a, b = cv2.split(roi_lab)
    
    features = []
    
    # Color features (HSV)
    features.extend([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v),  # Solar panels are DARK (low V)
    ])
    
    # LAB color features
    features.extend([
        np.mean(l), np.std(l),
        np.mean(a), np.std(a),
        np.mean(b), np.std(b),
    ])
    
    # Darkness features (key for solar panels!)
    features.extend([
        np.percentile(v, 25),  # 25th percentile darkness
        np.percentile(v, 50),  # Median darkness
        np.percentile(v, 75),  # 75th percentile
    ])
    
    # Texture features
    edges = cv2.Canny(roi_gray, 50, 150)
    features.extend([
        np.mean(edges),
        np.std(edges),
        np.sum(edges > 0) / edges.size,  # Edge density
    ])
    
    # Shape features
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    
    features.extend([
        width, height, area,
        aspect_ratio,
    ])
    
    # Blue color ratio (solar panels often have blue tint)
    # Use 'h' from HSV split (hue channel)
    blue_mask = ((h >= 90) & (h <= 130)).astype(np.uint8)
    blue_ratio = np.sum(blue_mask) / blue_mask.size
    features.append(blue_ratio)
    
    # Low saturation ratio (black panels)
    low_sat_mask = (s < 50).astype(np.uint8)
    low_sat_ratio = np.sum(low_sat_mask) / low_sat_mask.size
    features.append(low_sat_ratio)
    
    return features

def train_model(X, y):
    """Train Random Forest classifier"""
    print("\n" + "="*70)
    print("🤖 TRAINING SOLAR PANEL CLASSIFIER")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    clf.fit(X_scaled, y)
    
    # Calculate accuracy
    train_acc = clf.score(X_scaled, y)
    print(f"\n✅ Training accuracy: {train_acc*100:.1f}%")
    
    # Feature importance
    importances = clf.feature_importances_
    feature_names = [
        'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean', 'V_std',
        'L_mean', 'L_std', 'A_mean', 'A_std', 'B_mean', 'B_std',
        'V_25p', 'V_50p', 'V_75p',
        'Edge_mean', 'Edge_std', 'Edge_density',
        'Width', 'Height', 'Area', 'Aspect_ratio',
        'Blue_ratio', 'LowSat_ratio'
    ]
    
    print("\n📊 Top 5 most important features:")
    indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    return clf, scaler

def main():
    global img, original, current_label, solar_panels, non_panels
    
    print("="*70)
    print("☀️ SOLAR PANEL LABELING TOOL")
    print("="*70)
    
    print("\n⚠️  IMPORTANT: Label DARK BLUE/BLACK solar panels!")
    print("   Solar panels are distinctive by their dark color.")
    
    print("\n📋 Instructions:")
    print("   1. Draw rectangles around SOLAR PANELS (yellow)")
    print("   2. Press 'n' to switch to NON-PANELS mode (red)")
    print("   3. Label dark roofs, shadows, roads as NON-PANELS")
    print("   4. Press 's' to switch back to SOLAR PANELS mode")
    print("   5. Press 'u' to UNDO last label")
    print("   6. Press 'v' to SAVE training data")
    print("   7. Press 't' to TRAIN the model")
    print("   8. Press 'q' to QUIT")
    
    print("\n🎯 Goal: Label 20+ solar panels, 30+ non-panels")
    print("="*70)
    
    # Try to load existing model
    try:
        with open('solar_panel_classifier.pkl', 'rb') as f:
            clf, scaler = pickle.load(f)
        print("\n✅ Loaded existing model - you're ADDING to previous training")
        
        # Load previous labels
        try:
            with open('solar_training_labels.pkl', 'rb') as f:
                solar_panels, non_panels = pickle.load(f)
            print(f"✅ Loaded previous labels: {len(solar_panels)} solar panels, {len(non_panels)} non-panels")
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
    for idx, (x1, y1, x2, y2) in enumerate(solar_panels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"S{idx+1}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for idx, (x1, y1, x2, y2) in enumerate(non_panels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"N{idx+1}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.namedWindow('Label Solar Panels')
    cv2.setMouseCallback('Label Solar Panels', mouse_callback)
    
    print("\n✅ Window opened. Start labeling!")
    print(f"📊 Current: {len(solar_panels)} solar panels, {len(non_panels)} non-panels")
    
    while True:
        cv2.imshow('Label Solar Panels', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        
        elif key == ord('n'):
            current_label = 0
            print("\n🔴 Mode: NON-PANELS (label dark roofs, shadows, roads)")
        
        elif key == ord('s'):
            current_label = 1
            print("\n🟡 Mode: SOLAR PANELS (label solar panels)")
        
        elif key == ord('u'):
            # Undo last label
            img = original.copy()
            if current_label == 1 and solar_panels:
                solar_panels.pop()
                print(f"⏪ Removed last solar panel. Count: {len(solar_panels)}")
            elif current_label == 0 and non_panels:
                non_panels.pop()
                print(f"⏪ Removed last non-panel. Count: {len(non_panels)}")
            
            # Redraw all
            for idx, (x1, y1, x2, y2) in enumerate(solar_panels):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, f"S{idx+1}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            for idx, (x1, y1, x2, y2) in enumerate(non_panels):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"N{idx+1}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        elif key == ord('v'):
            # Save labels
            with open('solar_training_labels.pkl', 'wb') as f:
                pickle.dump((solar_panels, non_panels), f)
            print(f"\n💾 Saved {len(solar_panels)} solar panels, {len(non_panels)} non-panels")
        
        elif key == ord('t'):
            # Train model
            if len(solar_panels) < 5 or len(non_panels) < 5:
                print("\n⚠️  Need at least 5 solar panels and 5 non-panels to train!")
                continue
            
            print(f"\n🎓 Training with {len(solar_panels)} solar panels, {len(non_panels)} non-panels...")
            
            # Extract features
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            
            X = []
            y = []
            
            # Solar panels (positive class)
            for x1, y1, x2, y2 in solar_panels:
                features = extract_features(gray, hsv, lab, x1, y1, x2, y2)
                if features:
                    X.append(features)
                    y.append(1)
            
            # Non-panels (negative class)
            for x1, y1, x2, y2 in non_panels:
                features = extract_features(gray, hsv, lab, x1, y1, x2, y2)
                if features:
                    X.append(features)
                    y.append(0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train
            clf, scaler = train_model(X, y)
            
            # Save model
            with open('solar_panel_classifier.pkl', 'wb') as f:
                pickle.dump((clf, scaler), f)
            print("\n💾 Model saved to 'solar_panel_classifier.pkl'")
            
            # Save labels
            with open('solar_training_labels.pkl', 'wb') as f:
                pickle.dump((solar_panels, non_panels), f)
            print("💾 Labels saved to 'solar_training_labels.pkl'")
            
            print("\n✅ Training complete! Run detect_solar_panels.py to test.")
    
    cv2.destroyAllWindows()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()

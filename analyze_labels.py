import cv2
import pickle
import numpy as np

# Load labeled data
try:
    with open('solar_training_labels.pkl', 'rb') as f:
        solar_panels, non_panels = pickle.load(f)
    print(f"✅ Loaded {len(solar_panels)} solar panels, {len(non_panels)} non-panels")
except:
    print("❌ No training labels found!")
    exit()

# Load image
img = cv2.imread('temp_orthofoto.png')
if img is None:
    print("❌ No image found!")
    exit()

# Convert to HSV for analysis
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print("\n📊 SOLAR PANEL ANALYSIS:")
print("="*70)

if solar_panels:
    print(f"\n☀️ Analyzing {len(solar_panels)} labeled solar panels:")
    print("-"*70)
    print(" # | Position  | Size    | V-mean | V-min | V-max | S-mean | Dark?")
    print("-"*70)
    
    for idx, (x1, y1, x2, y2) in enumerate(solar_panels):
        roi_v = v[y1:y2, x1:x2]
        roi_s = s[y1:y2, x1:x2]
        
        v_mean = np.mean(roi_v)
        v_min = np.min(roi_v)
        v_max = np.max(roi_v)
        s_mean = np.mean(roi_s)
        
        w = x2 - x1
        h = y2 - y1
        
        is_dark = "✓ YES" if v_mean < 100 else "✗ NO"
        
        print(f"{idx+1:2d} | ({x1:4d},{y1:4d}) | {w:3d}×{h:3d} | {v_mean:6.1f} | {v_min:5.0f} | {v_max:5.0f} | {s_mean:6.1f} | {is_dark}")
    
    # Overall statistics
    all_v_means = [np.mean(v[y1:y2, x1:x2]) for x1, y1, x2, y2 in solar_panels]
    print("-"*70)
    print(f"Average V (brightness): {np.mean(all_v_means):.1f}")
    print(f"Range: {np.min(all_v_means):.1f} - {np.max(all_v_means):.1f}")
    
    dark_count = sum(1 for v_val in all_v_means if v_val < 80)
    print(f"\n🎯 Panels with V < 80 (very dark): {dark_count}/{len(solar_panels)}")
    
    if dark_count == 0:
        print("\n⚠️  WARNING: None of your labeled panels are very dark (V<80)!")
        print("   Solar panels should be much darker than regular roofs.")
        print("   You may have labeled regular roofs or light-colored objects.")
        print("   Try labeling ONLY the darkest blue/black rectangular areas.")

# Draw labeled regions
result = img.copy()
for x1, y1, x2, y2 in solar_panels:
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    roi_v = v[y1:y2, x1:x2]
    v_mean = np.mean(roi_v)
    cv2.putText(result, f"V:{v_mean:.0f}", (x1, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

for x1, y1, x2, y2 in non_panels:
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imwrite('debug_labeled_regions.png', result)
print(f"\n💾 Saved 'debug_labeled_regions.png' showing what you labeled")
print("   Yellow boxes = solar panels (with V brightness values)")
print("   Red boxes = non-panels")

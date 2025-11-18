"""
Analyze the actual labeled solar panels to understand their characteristics
"""
import cv2
import numpy as np
import pickle
import os

print("🔍 ANALYZING YOUR LABELED SOLAR PANELS")
print("="*80)

# Load the image
if not os.path.exists('temp_orthofoto.png'):
    print("❌ temp_orthofoto.png not found!")
    exit(1)

image = cv2.imread('temp_orthofoto.png')
print(f"📐 Image loaded: {image.shape[1]}x{image.shape[0]}")

# Load labeled data
if not os.path.exists('solar_training_labels.pkl'):
    print("❌ No training labels found! Please label some solar panels first.")
    exit(1)

with open('solar_training_labels.pkl', 'rb') as f:
    labels_data = pickle.load(f)

# Handle both dict and tuple formats
if isinstance(labels_data, dict):
    solar_regions = labels_data['solar']
    non_panel_regions = labels_data['non_panel']
else:
    solar_regions, non_panel_regions = labels_data

print(f"✅ Loaded {len(solar_regions)} labeled solar panels")
print(f"✅ Loaded {len(non_panel_regions)} labeled non-panels")
print()

# Analyze each labeled solar panel
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 30, 100)

print("☀️ DETAILED ANALYSIS OF YOUR LABELED SOLAR PANELS:")
print("="*80)
print(f"{'#':<4} {'Position':<12} {'Size':<10} {'Inner-V':<8} {'Border-V':<10} {'Diff':<6} {'Grid':<8} {'Frame?'}")
print("-"*80)

result_img = image.copy()

for idx, (x1, y1, x2, y2) in enumerate(solar_regions, 1):
    # Draw the labeled region
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
    cv2.putText(result_img, f"#{idx}", (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Extract region
    roi_v = hsv[y1:y2, x1:x2, 2]
    roi_edges = edges[y1:y2, x1:x2]
    
    w = x2 - x1
    h = y2 - y1
    area = w * h
    
    # Border analysis
    border_size = max(2, min(w, h) // 8)
    
    if w > border_size*4 and h > border_size*4:
        # Inner core
        inner_v = roi_v[border_size:-border_size, border_size:-border_size]
        
        # Borders
        top_border = roi_v[:border_size, :]
        bottom_border = roi_v[-border_size:, :]
        left_border = roi_v[:, :border_size]
        right_border = roi_v[:, -border_size:]
        
        inner_mean = np.mean(inner_v)
        border_mean = np.mean([np.mean(top_border), np.mean(bottom_border),
                              np.mean(left_border), np.mean(right_border)])
        
        diff = border_mean - inner_mean
    else:
        inner_mean = np.mean(roi_v)
        border_mean = np.mean(roi_v)
        diff = 0
    
    # Grid detection
    edge_density = np.sum(roi_edges > 0) / (w * h)
    
    # Determine if it has frame
    has_frame = diff > 10
    has_grid = edge_density > 0.05 and edge_density < 0.3
    
    frame_status = "✓ YES" if has_frame else "✗ NO"
    grid_status = f"{edge_density:.3f}"
    
    print(f"{idx:<4} ({x1:4},{y1:4}) {w:3}x{h:<3} {inner_mean:7.1f} {border_mean:9.1f} {diff:+6.1f} {grid_status:<8} {frame_status}")
    
    # Draw analysis on image
    text_y = y1 - 20
    cv2.putText(result_img, f"I:{inner_mean:.0f} B:{border_mean:.0f}", 
                (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

print()
print("="*80)
print("📊 SUMMARY:")
print(f"Total labeled panels: {len(solar_regions)}")
print()

# Statistics
all_inner = []
all_border = []
all_diffs = []

for x1, y1, x2, y2 in solar_regions:
    roi_v = hsv[y1:y2, x1:x2, 2]
    w = x2 - x1
    h = y2 - y1
    border_size = max(2, min(w, h) // 8)
    
    if w > border_size*4 and h > border_size*4:
        inner_v = roi_v[border_size:-border_size, border_size:-border_size]
        top_border = roi_v[:border_size, :]
        bottom_border = roi_v[-border_size:, :]
        left_border = roi_v[:, :border_size]
        right_border = roi_v[:, -border_size:]
        
        inner_mean = np.mean(inner_v)
        border_mean = np.mean([np.mean(top_border), np.mean(bottom_border),
                              np.mean(left_border), np.mean(right_border)])
        
        all_inner.append(inner_mean)
        all_border.append(border_mean)
        all_diffs.append(border_mean - inner_mean)

if all_diffs:
    print(f"Inner brightness range: {min(all_inner):.1f} - {max(all_inner):.1f} (avg: {np.mean(all_inner):.1f})")
    print(f"Border brightness range: {min(all_border):.1f} - {max(all_border):.1f} (avg: {np.mean(all_border):.1f})")
    print(f"Difference range: {min(all_diffs):+.1f} - {max(all_diffs):+.1f} (avg: {np.mean(all_diffs):+.1f})")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    avg_diff = np.mean(all_diffs)
    if avg_diff < -10:
        print(f"   ⚠️ Your panels have DARKER borders (avg diff: {avg_diff:.1f})")
        print(f"   → Need to detect: border < inner (opposite of current logic)")
    elif avg_diff < 10:
        print(f"   ⚠️ Your panels have similar border/inner brightness (avg diff: {avg_diff:.1f})")
        print(f"   → Frame detection won't work well (threshold is 10)")
        print(f"   → Consider lowering threshold to {max(5, avg_diff + 5):.0f}")
    else:
        print(f"   ✓ Your panels have lighter borders (avg diff: {avg_diff:.1f})")
        print(f"   → Current frame detection should work")

print()
print(f"💾 Saved: 'debug_labeled_analysis.png'")
cv2.imwrite('debug_labeled_analysis.png', result_img)

print()
print("✅ Analysis complete!")
print()
print("🔍 Next: Look at 'debug_labeled_analysis.png' to see your labeled panels")
print("   with their brightness values displayed.")

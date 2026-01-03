# Intelligent Scissors for Interactive Image Segmentation

Interactive object boundary tracing tool implementing the classic Intelligent Scissors algorithm using Dijkstra's shortest path with edge-based cost functions and real-time GUI preview.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 🎯 Overview

This implementation enables users to trace complex object boundaries by clicking seed points, with the algorithm automatically finding optimal paths along strong edges. The system combines efficient pathfinding with an intuitive interface for accurate foreground/background segmentation.

## ✨ Features

- **Multi-Feature Edge Detection**: Combines Sobel gradient magnitude and Laplacian with empirically tuned weights (0.43/0.43/0.14)
- **Dijkstra's Algorithm**: Optimized shortest path computation with 8-connectivity and priority queue (O(V log V) complexity)
- **Real-Time Preview**: Live path visualization as mouse moves between seed points
- **Interactive GUI**: Click-based workflow with undo, keyboard shortcuts, and visual feedback
- **Binary Mask Export**: Automated contour filling for clean foreground/background separation

---

## 🚀 Results

Successfully segmented diverse test images with varying edge complexity:

| Image | Seed Points | Edge Quality | Key Challenge |
|-------|-------------|--------------|---------------|
| **Bear** | 40 | Complex texture | Fur boundaries and textured background |
| **Balloon** | 8 | High contrast | Thin string attachment |
| **Plane** | 30 | Angled structures | Wing/tail geometry |
| **Lena** | 13 | Moderate contrast | Feathered hat and hair texture |

### Performance Metrics
- **Preprocessing**: 0.1–0.3s per image (one-time edge cost computation)
- **Real-Time Pathfinding**: <0.05s per mouse movement
- **Memory**: Linear scaling, handles up to 1024×1024 efficiently

---

## 🛠️ Tech Stack

**Core**: Python 3.7+ · OpenCV · NumPy  
**GUI**: Tkinter · PIL  
**Algorithms**: Dijkstra's shortest path · Sobel operators · Laplacian edge detection · heapq priority queue

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/masoud-rafiee/intelligent-scissors-segmentation.git
cd intelligent-scissors-segmentation

# Install dependencies
pip install opencv-python numpy Pillow

# Run application
python main.py
```

**Requirements**: Python 3.7+, opencv-python, numpy, Pillow (tkinter included with Python)

---

## 💡 Usage

### Workflow
1. **Load Image**: Click "Load Image" and select a grayscale image
2. **Place Seeds**: Left-click to add seed points along object boundary
3. **Preview Path**: Move mouse to see real-time path suggestions
4. **Confirm Points**: Click to lock in the current path segment
5. **Close Contour**: Press `C` when boundary is complete
6. **Save Result**: Press `S` to export binary mask

### Controls
| Action | Input |
|--------|-------|
| Add seed point | Left click |
| Undo last point | Right click |
| Close contour | `C` key |
| Reset all | `R` key |
| Save result | `S` key |

### Tips
- Start with high-contrast test images (balloon.pgm recommended)
- Place seed points at corners and curves for best results
- Use real-time preview to validate path before confirming
- Ensure contour is fully closed before saving

---

## 📂 Project Structure

```
intelligent-scissors-segmentation/
├── main.py                    # Application entry point
├── intelligent_scissors.py    # Core algorithm implementation
├── README.md
├── docs/
│   └── Intelligent-Scissors-Implementation-Report.pdf
├── test_images/               # Sample PGM images
│   ├── bear.pgm
│   ├── balloon.pgm
│   ├── plane.pgm
│   └── lena.pgm
└── results/                   # Output binary masks
```

---

## 🔬 Implementation Details

### Edge Cost Function
```python
cost = 0.43 * (1 - gradient_magnitude) + 
       0.43 * (1 - laplacian) + 
       0.14
```
Lower costs indicate stronger edges that the algorithm prefers when finding paths.

### Algorithm Flow
1. **Precompute Edge Costs**: Sobel gradients + Laplacian on full image
2. **User Interaction**: Click seed points, system shows live path preview
3. **Dijkstra Pathfinding**: Find minimum-cost path from last seed to cursor position
4. **Path Confirmation**: Lock in path segment on click
5. **Mask Generation**: Fill closed contour to create binary segmentation

### Key Optimizations
- 8-connectivity neighborhoods for smooth curved boundaries
- Priority queue (heapq) for efficient minimum-cost node selection
- One-time edge cost computation with real-time path queries
- Parent pointer tracking for fast path reconstruction

---

## 📄 Documentation

Full implementation report with methodology, cost function tuning, performance analysis, and detailed results: [docs/Intelligent-Scissors-Implementation-Report.pdf](docs/Intelligent-Scissors-Implementation-Report.pdf)

---

## 🐛 Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| PGM format loading fails | Fallback to PIL with NumPy array conversion |
| Jagged boundaries | Use 8-connectivity instead of 4-connectivity |
| Suboptimal paths | Tune cost weights (current: 0.43/0.43/0.14) |

---

## 🔮 Future Enhancements

- Magnetic lasso with automatic seed suggestion
- Multi-object segmentation support
- Machine learning-based edge cost refinement
- Vector format export (SVG/EPS)
- Color image support with channel weighting

---

## 📚 References

- Mortensen & Barrett (1995). *Intelligent scissors for image composition*. SIGGRAPH.
- OpenCV Documentation: https://docs.opencv.org
- Course: CS563 Computer Vision, Fall 2025

---

## 👥 Contributors

**Team ReSpawn**: [Masoud Rafiee](https://github.com/masoud-rafiee), Sonia Tayeb Cherif

---

## 📝 License

This project is available under the MIT License. See LICENSE file for details.

---

<div align="center">

**Built with** 🖱️ **Interactive CV** · 🎯 **Dijkstra's Algorithm** · 🖼️ **Edge Detection**

</div>
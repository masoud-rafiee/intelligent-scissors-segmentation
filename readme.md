\# Intelligent Scissors - CS563 Assignment 3



Interactive image segmentation tool using the Intelligent Scissors algorithm (live-wire boundary detection).



\## Team Members

\- Masoud Rafiee

\- Sonia Tayeb Cherif



\## Requirements



```bash

pip install opencv-python numpy Pillow

```



\*\*Python Version:\*\* 3.7 or higher



\## Quick Start



1\. \*\*Run the program:\*\*

&nbsp;  ```bash

&nbsp;  python main.py

&nbsp;  ```



2\. \*\*Load an image\*\* from the `Images/` folder (balloon.pgm, bear.pgm, lena.pgm, plane.pgm)



3\. \*\*Segment the object:\*\*

&nbsp;  - Click to place seed points around object boundary

&nbsp;  - Watch the yellow preview path as you move the mouse

&nbsp;  - Click to confirm each point (turns green)

&nbsp;  - Continue until you complete the contour



4\. \*\*Save the result:\*\*

&nbsp;  - Press `C` to close the contour

&nbsp;  - Press `S` to save the binary mask



\## Controls



| Action | Control |

|--------|---------|

| Add seed point | Left Click |

| Undo last point | Right Click |

| Close contour | `C` key or button |

| Reset all | `R` key or button |

| Save result | `S` key or button |



\## Features



✓ Real-time path preview (live-wire feedback)  

✓ Dijkstra's shortest path algorithm  

✓ Edge detection using gradient + Laplacian  

✓ Interactive GUI with visual feedback  

✓ Binary mask output (foreground/background)  

✓ Support for PGM, PNG, JPG formats  



\## Project Structure



```

a3/

├── main.py                    # Main program

├── Images/                    # Test images folder

│   ├── balloon.pgm

│   ├── bear.pgm

│   ├── lena.pgm

│   └── plane.pgm

├── README.md                  # This file

└── Assignment3\_Report.pdf     # Detailed report

```



\## How It Works



1\. \*\*Edge Cost Calculation:\*\* Combines gradient magnitude and Laplacian to identify strong edges

2\. \*\*Path Finding:\*\* Dijkstra's algorithm finds minimum cost path between points

3\. \*\*Real-time Feedback:\*\* Yellow path shows optimal route before confirming

4\. \*\*Contour Closure:\*\* Connects last point to first to create closed boundary

5\. \*\*Mask Generation:\*\* Fills closed contour to create binary segmentation



\## Tips



\- Start with high-contrast images for easier segmentation

\- Place seed points at corners and areas of high curvature

\- Use the live preview to ensure the path follows the edge correctly

\- Right-click immediately if a point is misplaced

\- Ensure the contour is fully closed before saving



\## Output



The program generates a binary image where:

\- \*\*White (255)\*\* = Foreground (segmented object)

\- \*\*Black (0)\*\* = Background



\## Troubleshooting



\*\*Image won't load:\*\*

\- Ensure the image is in the `Images/` folder

\- Check file format (PGM, PNG, JPG supported)



\*\*Path doesn't follow edge:\*\*

\- Place more seed points in problematic areas

\- Try adjusting point placement slightly



\*\*Save button disabled:\*\*

\- Make sure to close the contour first (press `C`)



\## Algorithm Details



\- \*\*Edge Cost Function:\*\* 0.43×(1-gradient) + 0.43×(1-laplacian) + 0.14

\- \*\*Connectivity:\*\* 8-neighborhood (diagonal movement allowed)

\- \*\*Complexity:\*\* O((V+E)log V) where V = pixels, E = edges



\## License



Educational project for CS563-463 Winter 2025



\## Contact



For questions or issues, contact team members via course communication channels.


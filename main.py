"""
Intelligent Scissors Implementation
CS563-463 Winter 2025 - Computer Vision
Assignment 3
Masoud Rafiee
Sonia Tayeb Cherif
This program implements the Intelligent Scissors algorithm for interactive
image segmentation using live-wire boundary detection.
"""
import cv2
import numpy as np
import heapq
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class IntelligentScissors:
    """
    Intelligent Scissors algorithm implementation.
    Uses Dijkstra's algorithm with edge cost function based on:
    - Gradient magnitude
    - Gradient direction
    - Laplacian zero-crossing
    """

    def __init__(self, image):
        """
        Initialize with grayscale image.

        Args:
            image: Grayscale numpy array (H x W)
        """
        self.image = image
        self.height, self.width = image.shape

        # Precompute edge costs
        self._compute_edge_costs()

    def _compute_edge_costs(self):
        """
        Precompute edge costs based on image gradients.
        Lower cost = stronger edge = more likely boundary.
        """
        # Convert to float for gradient computation
        img_float = self.image.astype(np.float32)

        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        self.gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient magnitude to [0, 1]
        max_grad = np.max(self.gradient_mag)
        if max_grad > 0:
            self.gradient_mag /= max_grad

        # Compute Laplacian for zero-crossing detection
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=3)
        self.laplacian = np.abs(laplacian)

        # Normalize Laplacian
        max_lap = np.max(self.laplacian)
        if max_lap > 0:
            self.laplacian /= max_lap

        # Gradient direction
        self.grad_direction = np.arctan2(grad_y, grad_x)

    def _get_edge_cost(self, y1, x1, y2, x2):
        """
        Calculate edge cost between two adjacent pixels.

        Args:
            y1, x1: Coordinates of first pixel
            y2, x2: Coordinates of second pixel

        Returns:
            Edge cost (lower = stronger edge)
        """
        # Edge cost based on inverted gradient magnitude
        # Strong edges have low cost
        grad_cost = 1.0 - self.gradient_mag[y2, x2]

        # Add Laplacian component
        lap_cost = 1.0 - self.laplacian[y2, x2]

        # Weighted combination
        cost = 0.43 * grad_cost + 0.43 * lap_cost + 0.14

        return cost

    def find_path(self, start, end):
        """
        Find minimum cost path from start to end using Dijkstra's algorithm.

        Args:
            start: (x, y) tuple of start point
            end: (x, y) tuple of end point

        Returns:
            List of (x, y) coordinates forming the path
        """
        start_x, start_y = start
        end_x, end_y = end

        # Check bounds
        if not (0 <= start_x < self.width and 0 <= start_y < self.height):
            return []
        if not (0 <= end_x < self.width and 0 <= end_y < self.height):
            return []

        # Dijkstra's algorithm
        # Priority queue: (cost, y, x)
        pq = [(0, start_y, start_x)]
        visited = set()
        cost_map = {(start_y, start_x): 0}
        parent_map = {}

        # 8-connectivity neighborhood
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 1),
                     (1, -1), (1, 0), (1, 1)]

        while pq:
            current_cost, cy, cx = heapq.heappop(pq)

            # Found destination
            if (cy, cx) == (end_y, end_x):
                break

            # Skip if already visited
            if (cy, cx) in visited:
                continue

            visited.add((cy, cx))

            # Explore neighbors
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx

                # Check bounds
                if not (0 <= ny < self.height and 0 <= nx < self.width):
                    continue

                # Skip if visited
                if (ny, nx) in visited:
                    continue

                # Calculate edge cost
                edge_cost = self._get_edge_cost(cy, cx, ny, nx)
                new_cost = current_cost + edge_cost

                # Update if better path found
                if (ny, nx) not in cost_map or new_cost < cost_map[(ny, nx)]:
                    cost_map[(ny, nx)] = new_cost
                    parent_map[(ny, nx)] = (cy, cx)
                    heapq.heappush(pq, (new_cost, ny, nx))

        # Reconstruct path
        if (end_y, end_x) not in parent_map and (end_y, end_x) != (start_y, start_x):
            # No path found, return straight line
            return self._bresenham_line(start_x, start_y, end_x, end_y)

        path = []
        current = (end_y, end_x)

        while current in parent_map:
            path.append((current[1], current[0]))  # Convert to (x, y)
            current = parent_map[current]

        path.append((start_x, start_y))
        path.reverse()

        return path

    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Generate line using Bresenham's algorithm (fallback).

        Returns:
            List of (x, y) coordinates
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points


class IntelligentScissorsGUI:
    """
    Interactive GUI for Intelligent Scissors application.
    """

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Intelligent Scissors - CS563 Assignment 3")
        self.root.geometry("1200x800")

        # State variables
        self.image = None
        self.display_image = None
        self.scissors = None
        self.points = []  # List of seed/free points
        self.paths = []  # List of paths between consecutive points
        self.temp_path = []  # Temporary path while moving mouse
        self.closed = False

        # Display parameters
        self.scale = 1.0
        self.canvas_image = None

        self._setup_ui()
        self._show_greeting()

    def _setup_ui(self):
        """Setup the user interface."""
        # Control panel
        control_frame = tk.Frame(self.root, bg='#2b2b2b', padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Title
        title = tk.Label(control_frame, text="Intelligent Scissors",
                         font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        title.pack(pady=10)

        # Load image button
        self.load_btn = tk.Button(control_frame, text="Load Image",
                                  command=self._load_image,
                                  bg='#4CAF50', fg='white',
                                  font=('Arial', 12), padx=20, pady=10)
        self.load_btn.pack(pady=5, fill=tk.X)

        # Instructions
        instructions = tk.Label(control_frame,
                                text="Instructions:\n\n"
                                     "1. Load an image\n"
                                     "2. Click to place seed points\n"
                                     "3. Move mouse to see live path\n"
                                     "4. Click to confirm point\n"
                                     "5. Close contour or finish\n\n"
                                     "Controls:\n"
                                     "• Left Click: Add point\n"
                                     "• Right Click: Undo point\n"
                                     "• 'C': Close contour\n"
                                     "• 'R': Reset all\n"
                                     "• 'S': Save result",
                                font=('Arial', 10),
                                fg='white', bg='#2b2b2b',
                                justify=tk.LEFT)
        instructions.pack(pady=20, padx=10)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(control_frame,
                                textvariable=self.status_var,
                                font=('Arial', 10),
                                fg='#4CAF50', bg='#2b2b2b',
                                wraplength=200)
        status_label.pack(pady=10)

        # Action buttons
        self.close_btn = tk.Button(control_frame, text="Close Contour (C)",
                                   command=self._close_contour,
                                   bg='#2196F3', fg='white',
                                   font=('Arial', 10), padx=15, pady=8,
                                   state=tk.DISABLED)
        self.close_btn.pack(pady=5, fill=tk.X)

        self.reset_btn = tk.Button(control_frame, text="Reset (R)",
                                   command=self._reset,
                                   bg='#FF9800', fg='white',
                                   font=('Arial', 10), padx=15, pady=8,
                                   state=tk.DISABLED)
        self.reset_btn.pack(pady=5, fill=tk.X)

        self.save_btn = tk.Button(control_frame, text="Save Result (S)",
                                  command=self._save_result,
                                  bg='#9C27B0', fg='white',
                                  font=('Arial', 10), padx=15, pady=8,
                                  state=tk.DISABLED)
        self.save_btn.pack(pady=5, fill=tk.X)

        # Canvas for image display
        canvas_frame = tk.Frame(self.root, bg='#1e1e1e')
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e',
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.root.bind('<c>', lambda e: self._close_contour())
        self.root.bind('<r>', lambda e: self._reset())
        self.root.bind('<s>', lambda e: self._save_result())

    def _show_greeting(self):
        """Display greeting message."""
        greeting = (
            "Welcome to Intelligent Scissors!\n\n"
            "This program implements the intelligent scissors algorithm\n"
            "for interactive image segmentation.\n\n"
            "Click 'Load Image' to begin."
        )
        messagebox.showinfo("Welcome", greeting)

    def _load_image(self):
        """Load image from file."""
        # Look for Images folder first
        initial_dir = Path("Images")
        if not initial_dir.exists():
            initial_dir = Path.cwd()

        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=str(initial_dir),
            filetypes=[("PGM files", "*.pgm"),
                       ("PNG files", "*.png"),
                       ("JPEG files", "*.jpg *.jpeg"),
                       ("All files", "*.*")]
        )

        if not file_path:
            return

        # Load image
        try:
            # Try OpenCV first
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Try PIL for PGM files
                pil_img = Image.open(file_path).convert('L')
                img = np.array(pil_img)

            if img is None:
                raise ValueError("Could not load image")

            self.image = img
            self.scissors = IntelligentScissors(img)

            # Reset state
            self._reset()

            # Display image
            self._display_image()

            # Enable buttons
            self.reset_btn.config(state=tk.NORMAL)
            self.close_btn.config(state=tk.NORMAL)

            self.status_var.set(f"Loaded: {Path(file_path).name}\n"
                                f"Size: {img.shape[1]}x{img.shape[0]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def _display_image(self):
        """Display image on canvas with paths."""
        if self.image is None:
            return

        # Create color version for display
        display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        # Draw confirmed paths in green
        for path in self.paths:
            for i in range(len(path) - 1):
                cv2.line(display, path[i], path[i + 1], (0, 255, 0), 2)

        # Draw temporary path in yellow
        if self.temp_path and len(self.temp_path) > 1:
            for i in range(len(self.temp_path) - 1):
                cv2.line(display, self.temp_path[i], self.temp_path[i + 1],
                         (255, 255, 0), 2)

        # Draw points
        for i, point in enumerate(self.points):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)  # Red for first, blue for others
            cv2.circle(display, point, 5, color, -1)
            cv2.circle(display, point, 6, (255, 255, 255), 2)

        # Convert to PhotoImage
        self.display_image = display

        # Scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale_w = canvas_width / display.shape[1]
            scale_h = canvas_height / display.shape[0]
            self.scale = min(scale_w, scale_h, 1.0)

            new_width = int(display.shape[1] * self.scale)
            new_height = int(display.shape[0] * self.scale)

            display_resized = cv2.resize(display, (new_width, new_height))
        else:
            display_resized = display

        # Convert to PIL Image
        pil_image = Image.fromarray(display_resized)
        self.photo = ImageTk.PhotoImage(pil_image)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas_image = self.photo

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates."""
        img_x = int(canvas_x / self.scale)
        img_y = int(canvas_y / self.scale)
        return (img_x, img_y)

    def _on_click(self, event):
        """Handle left mouse click - add point."""
        if self.image is None or self.closed:
            return

        # Convert to image coordinates
        point = self._canvas_to_image_coords(event.x, event.y)

        # Check bounds
        if not (0 <= point[0] < self.image.shape[1] and
                0 <= point[1] < self.image.shape[0]):
            return

        # Add point
        self.points.append(point)

        # If we have at least 2 points, compute path
        if len(self.points) >= 2:
            path = self.scissors.find_path(self.points[-2], self.points[-1])
            self.paths.append(path)

        # Clear temp path
        self.temp_path = []

        # Update display
        self._display_image()

        # Update status
        self.status_var.set(f"Points: {len(self.points)}\n"
                            f"Click to add more points\n"
                            f"or press 'C' to close")

        # Enable save button if contour is closed
        if self.closed:
            self.save_btn.config(state=tk.NORMAL)

    def _on_right_click(self, event):
        """Handle right mouse click - undo last point."""
        if self.image is None or len(self.points) == 0 or self.closed:
            return

        # Remove last point and path
        self.points.pop()
        if len(self.paths) > 0:
            self.paths.pop()

        self.temp_path = []

        # Update display
        self._display_image()

        self.status_var.set(f"Points: {len(self.points)}\n"
                            f"Removed last point")

    def _on_mouse_move(self, event):
        """Handle mouse movement - show live path."""
        if self.image is None or len(self.points) == 0 or self.closed:
            return

        # Convert to image coordinates
        current_point = self._canvas_to_image_coords(event.x, event.y)

        # Check bounds
        if not (0 <= current_point[0] < self.image.shape[1] and
                0 <= current_point[1] < self.image.shape[0]):
            return

        # Compute path from last point to current mouse position
        self.temp_path = self.scissors.find_path(self.points[-1], current_point)

        # Update display
        self._display_image()

    def _close_contour(self):
        """Close the contour by connecting last point to first."""
        if self.image is None or len(self.points) < 3:
            messagebox.showwarning("Warning",
                                   "Need at least 3 points to close contour")
            return

        if self.closed:
            return

        # Connect last point to first
        path = self.scissors.find_path(self.points[-1], self.points[0])
        self.paths.append(path)

        self.closed = True
        self.temp_path = []

        # Update display
        self._display_image()

        # Enable save button
        self.save_btn.config(state=tk.NORMAL)
        self.status_var.set("Contour closed!\nClick 'Save Result' to export")

    def _reset(self):
        """Reset all points and paths."""
        self.points = []
        self.paths = []
        self.temp_path = []
        self.closed = False

        if self.image is not None:
            self._display_image()
            self.status_var.set("Reset complete\nClick to add points")

        self.save_btn.config(state=tk.DISABLED)

    def _save_result(self):
        """Save binary segmentation result."""
        if self.image is None or not self.closed:
            messagebox.showwarning("Warning",
                                   "Please close the contour first")
            return

        # Create binary mask
        mask = np.zeros(self.image.shape, dtype=np.uint8)

        # Draw all paths
        for path in self.paths:
            points_array = np.array(path, dtype=np.int32)
            cv2.polylines(mask, [points_array], False, 255, 1)

        # Fill the contour
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Fill the largest contour
            cv2.drawContours(mask, contours, -1, 255, -1)

        # Save file
        file_path = filedialog.asksaveasfilename(
            title="Save Binary Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("PGM files", "*.pgm"),
                       ("All files", "*.*")]
        )

        if file_path:
            cv2.imwrite(file_path, mask)
            messagebox.showinfo("Success",
                                f"Binary result saved to:\n{file_path}")
            self.status_var.set("Result saved successfully!")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = IntelligentScissorsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
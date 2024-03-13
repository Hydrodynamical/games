import tkinter as tk

def highlight_canvas(event):
    # Set the highlight color to blue and the thickness to 2 when the canvas is clicked
    event.widget.config(highlightbackground='blue', highlightcolor='blue', highlightthickness=2)

# Create the main window
root = tk.Tk()
root.title("Highlight Canvas Example")

# Create a canvas widget with a thicker border ready for highlighting
canvas = tk.Canvas(root, width=200, height=100, bd=0, highlightthickness=1, highlightbackground='white')
canvas.pack(padx=10, pady=10)

# Bind the left mouse button click event to the highlight function
canvas.bind("<Button-1>", highlight_canvas)

root.mainloop()

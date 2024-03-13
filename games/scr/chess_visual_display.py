# Import the class GameState from the module chess_engine
from chess_engine import GameState

# Testing visuals for Tkinter
import tkinter as tk

def on_button_click():
    print("Button clicked!")

root = tk.Tk()
root.title("Tkinter Example")

button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack()

root.mainloop()

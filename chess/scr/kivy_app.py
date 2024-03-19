from kivy.app import App
from kivy.uix.button import Button

class TestApp(App):
    def build(self):
        # Create a button with the text 'Press Me'
        self.button = Button(text='Press Me')
        
        # Bind the button's on_press event to the callback method self.on_button_press
        self.button.bind(on_press=self.on_button_press)
        
        return self.button

    def on_button_press(self, instance):
        # Change the button text when it's pressed
        instance.text = 'Hello, World!'

if __name__ == '__main__':
    TestApp().run()

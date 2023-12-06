from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.toast import toast

from generator import Generator


class GUI(MDApp):
    progress_value = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_file_name = "Null"
        self.screen = Builder.load_file('./gui.kv')
        self.hidden_layers = None

    def build(self):
        self.title = 'Artificial Neural Network'
        self.icon = './icon.ico'
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        Window.size = (640, 480)
        Window.bind(on_dropfile=self.on_file_drop)
        return self.screen

    def on_file_drop(self, window, file_path):
        self.source_file_name = file_path.decode("utf-8")

    def generate(self):
        if self.source_file_name != "Null":
            if self.root.ids.layer_count.text != "":
                self.hidden_layers = []
                text = self.root.ids.layer_count.text
                layers = text.split(",")
                flag = 0
                for layer in layers:
                    try:
                        self.hidden_layers.append(int(layer))
                    except ValueError:
                        flag = 1
                        toast("Sayı girin")
                if flag == 0:
                    generator = Generator(self.hidden_layers, self.source_file_name)
                    generator.start()
            else:
                toast("Katman boş olamaz")
        else:
            toast("Bir dosya seçmediniz")


if __name__ == '__main__':
    GUI().run()

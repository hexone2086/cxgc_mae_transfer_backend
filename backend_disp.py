import PIL.Image
import PIL.ImageTk
import cv2
import threading
import time
import PIL
from loguru import logger
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar

from backend_utils import infer_output

class GUI:
    def __init__(self, worker = None):
        self.worker = worker
        self.worker_thread = None

        self.player_thread = None

        self.window = Tk()
        self.window.geometry("640x480")
        self.window.configure(bg = "#FFFFFF")

        self.canvas = Canvas(
            self.window,
            bg = "#FFFFFF",
            height = 480,
            width = 640,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)


        self.image_image_1 = PhotoImage(
            file="./assets_ui/image_1.png")
        image_1 = self.canvas.create_image(
            320.0,
            240.0,
            image=self.image_image_1
        )

        self.canvas.create_rectangle(
            433.0,
            40.0,
            622.0,
            264.0,
            fill="#B0B5EC",
            outline="")

        self.canvas.create_rectangle(
            15.0,
            35.0,
            425.0,
            445.0,
            fill="#B0B5EC",
            outline="")

        self.image_image_2 = PhotoImage(
            file="./assets_ui/image_2.png")
        image_2 = self.canvas.create_image(
            220.0,
            240.0,
            image=self.image_image_2
        )
        
        self.canvas.create_text(
            440.0,
            63.0,
            anchor="nw",
            text="real time bitrate",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        self.canvas.create_text(
            440.0,
            134.0,
            anchor="nw",
            text="mask ratio",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        self.canvas.create_text(
            440.0,
            90.0,
            anchor="nw",
            text="xxxxx",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        self.canvas.create_text(
            440.0,
            160.0,
            anchor="nw",
            text="xxxxx",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        self.canvas.create_text(
            530.0,
            90.0,
            anchor="nw",
            text="kbps",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        self.canvas.create_text(
            530.0,
            160.0,
            anchor="nw",
            text="%",
            fill="#000000",
            font=("Inter", 20 * -1)
        )

        # 初始化界面元素
        self.entry_var = StringVar()
        self.entry_1 = Entry(
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            textvariable=self.entry_var
        )
        self.entry_1.place(
            x=433.0,
            y=277.0,
            width=189.0,
            height=39.0,
        )

        # # 设置提示文本
        self.entry_var.set("Enter text here")
        self.entry_1.bind("<FocusIn>", self.on_entry_focus_in)
        self.entry_1.bind("<FocusOut>", self.on_entry_focus_out)

        self.button_image_1 = PhotoImage(
            file="./assets_ui/button_1.png")
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.button_1_click,
            relief="flat"
        )
        self.button_1.place(
            x=433.0,
            y=331.0,
            width=189.0,
            height=109.0
        )

        self.window.resizable(False, False)

    def on_entry_focus_in(self, event):
        if self.entry_var.get() == "enter remote address":
            self.entry_var.set("")

    def on_entry_focus_out(self, event):
        if self.entry_var.get() == "":
            self.entry_var.set("enter remote address")

    def button_1_click(self):
        logger.debug("button_1 clicked")
        if self.worker is not None:
            if self.worker_thread is None:
                self.worker_thread = threading.Thread(target=self.worker)
                self.player_thread = threading.Thread(target=self.player)
            if self.worker_thread.is_alive():
                # TODO 增加线程退出接口
                logger.warning("[unavailable] kill mae_infer worker")
            else:
                logger.info("start mae_infer worker")
                self.worker_thread.start()
                self.player_thread.start()
            # logger.info("start mae_infer worker")
            # self.infer_thread.start()

    def draw_frame(self, frame):
        frame = cv2.resize(frame, (400, 400))
        self.image_frame = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame))

        _image_frame = self.canvas.create_image(
            220.0,
            240.0,
            image=self.image_frame
        )

    def player(self):
        # TODO 从图像序列中获取图像，并更新到 image_2 中
        while True:
            if len(infer_output) > 0:
                frame = infer_output.popleft()
                self.draw_frame(frame)
                time.sleep(0.1)
            pass
if __name__ == '__main__':
    gui = GUI()
    gui.window.mainloop()
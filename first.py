from tkinter import *
window=Tk()
window.geometry("600x1200")
label1=Label(window, text="First label",fg="#333",bg="#fff", width="20", font=("", 20))
label1.grid(row="0",column="0")
label2=Label(window, width="40",bg="#008080")
label2.grid(row="0",column="1")
window.mainloop()
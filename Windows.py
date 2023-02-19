from tkinter import *
from tkinter import messagebox

# Funcion para el boton
def showMessage():
    messagebox.showinfo(
        title='Ventana informativa',
        message='Estas en el taller de Python'
    )

window=Tk()
window.title('Taller de Python')
# window.geometry('400x400')
window.configure(bg='green')
window.state('zoomed')

button1=Button(window, text='Aceptar', command=showMessage)
button1.place(x=20, y=20)

window.mainloop()
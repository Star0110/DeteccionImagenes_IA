from tkinter import filedialog
from prediccion_imagen import predict_image

path=filedialog.askopenfilename(title="Selecciona un archivo")
predict_image(path)

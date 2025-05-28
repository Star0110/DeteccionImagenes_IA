from tkinter import filedialog
from prediccion_imagen import predict_image,predict_video

path=filedialog.askopenfilename(title="Selecciona un archivo")
#predict_image(path)
predict_video(path)
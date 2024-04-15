import gradio as gr
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
from models.define_LSTM5 import define_LSTM5

# import LSTM5 model
data_shape = (875, 1)
model = define_LSTM5(data_shape)
model.load_weights('D:/finalproject/outcomes/LSTM5.result_retrain_pers_random_cb.h5')  # Replace with the real path to the model weights file


def plot_rppg_and_bp(index):
    data_file = 'D:/finalproject/data/rPPG-BP-UKL_rppg_7s.h5'
    with h5py.File(data_file, 'r') as f:
        rppg = f.get('rppg')
        rppg = np.transpose(np.array(rppg), axes=(1, 0))
        rppg = np.expand_dims(rppg, axis=2)
        bp_labels = f.get('label')
        bp_labels = np.array(bp_labels)


    rppg_data = rppg[index, :, 0]
    rppg_sample = np.expand_dims(rppg_data, axis=0)

    # Use model to predict blood pressure values
    predicted_bp = model.predict(rppg_sample)
    predicted_sbp = round(predicted_bp[0][0][0])
    predicted_dbp = round(predicted_bp[1][0][0])

    # Plotting
    time_axis = np.linspace(0, 7, len(rppg_data))
    fig, ax = plt.subplots()
    ax.plot(time_axis, rppg_data)
    ax.set_title(f'Visualization of rPPG Data at Index {index}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('rPPG Signal Value')
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    sbp = round(bp_labels[0, index])
    dbp = round(bp_labels[1, index])
    bp_text = f"SBP: {sbp} mmHg, DBP: {dbp} mmHg"
    model_bp_text = f"Predicted SBP: {predicted_sbp} mmHg, Predicted DBP: {predicted_dbp} mmHg"

    return img, bp_text, model_bp_text


interface = gr.Interface(
    fn=plot_rppg_and_bp,
    inputs=gr.Slider(minimum=0, maximum=7851, value=0, label="Index of rPPG Signal"),
    outputs=["image", "text", "text"],
    title="rPPG Signal and Blood Pressure Visualizer",
    description="Move the slider to view different rPPG signals and corresponding blood pressure values, including model predictions."
)

interface.launch()








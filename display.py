import gradio as gr
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def plot_rppg_and_bp(index):
    data_file = 'D:/finalproject/data/rPPG-BP-UKL_rppg_7s.h5'
    with h5py.File(data_file, 'r') as f:
        rppg = f.get('rppg')
        rppg = np.transpose(np.array(rppg), axes=(1, 0))
        rppg = np.expand_dims(rppg, axis=2)
        bp_labels = f.get('label')  # 假设label数据是SBP和DBP的值
        bp_labels = np.array(bp_labels)

    # 获取指定索引的rPPG数据和血压标签
    rppg_data = rppg[index, :, 0]
    time_axis = np.linspace(0, 7, len(rppg_data))
    sbp = round(bp_labels[0, index])  # 取整
    dbp = round(bp_labels[1, index])  # 取整

    # 绘制rPPG图形
    fig, ax = plt.subplots()
    ax.plot(time_axis, rppg_data)
    ax.set_title(f'Visualization of rPPG Data at Index {index}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('rPPG Signal Value')
    ax.grid(True)

    # 将matplotlib图转换为PIL Image对象返回
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # 组合血压值为文本输出
    bp_text = f"SBP: {sbp} mmHg, DBP: {dbp} mmHg"

    return img, bp_text


# 设置Gradio界面
interface = gr.Interface(
    fn=plot_rppg_and_bp,
    inputs=gr.Slider(minimum=0, maximum=7851, value=0, label="Index of rPPG Signal"),
    outputs=["image", "text"],
    title="rPPG Signal and Blood Pressure Visualizer",
    description="Move the slider to view different rPPG signals and corresponding blood pressure values."
)

interface.launch()


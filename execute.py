from retrain_rppg_personalization import retrain_rppg_personalization

# 设置参数
data_file = "D:/finalproject/data/rPPG-BP-UKL_rppg_7s.h5"
model_file = "D:/finalproject/LSTM5_ppg_nonmixed.h5"
experiment_name = "LSTM5.result"
checkpoint_path = "D:/finalproject/outcomes"
results_path = "D:/finalproject/outcomes"

# 调用主函数
retrain_rppg_personalization(data_file, model_file, experiment_name, checkpoint_path, results_path)


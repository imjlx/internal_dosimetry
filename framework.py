
import tensorflow as tf
import datetime
import time
import os
from tqdm import tqdm

from utils import data, loss, metrics, visual
from model import AutoEncoder


class TrainFramework(object):
    def __init__(self, model, p_ids, batch):
        # 设置使用的网络模型
        self.model = model
        # 读取数据集
        self.ds = data.create_train_dataset(p_ids, batch)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        # 检查点
        self.CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints/" + self.CURRENT_TIME
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        # tensorboard
        train_log_dir = "logs/" + self.CURRENT_TIME + "/train"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def train_step(self, ct, pet, source, dosemap):
        with tf.GradientTape() as g:
            dosemap = self.model(inputs=[ct, pet, source], training=True)






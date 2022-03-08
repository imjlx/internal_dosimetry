
import tensorflow as tf
import datetime
import time
import os
from tqdm import tqdm

from utils import Data
from utils import Loss, Metrics, Visual
from model import AutoEncoder


class TrainFramework(object):
    def __init__(self, model, p_ids, batch):
        # 设置使用的网络模型
        self.model = model
        # 读取数据集
        self.train_ds = data.create_train_dataset(p_ids, batch)
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

        self.save_model_per_n_epoch = 50

    @tf.function
    def train_step(self, ct, pet, source, dosemap):
        with tf.GradientTape() as g:
            dosemap_pred = self.model(inputs=[ct, pet, source], training=True)
            loss = Loss.total_loss(dosemap, dosemap_pred)
        gradient = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return loss

    def test_step(self):
        pass

    def fit(self, epochs):
        print("Start Time: ", self.CURRENT_TIME)
        time.sleep(0.01)
        start = time.time()

        for epoch in range(epochs):
            """
                训练
            """
            # loss保存一轮中每一批的损失函数
            loss = []
            with tqdm(self.train_ds.enumerate()) as bar:
                bar.set_description("Epoch %i" % epoch)
                for n, (ct, pet, source, dosemap) in bar:
                    loss_add = self.train_step(ct, pet, source, dosemap)
                    loss.append(loss_add)
            # 对所有的批求平均，得到本轮的平均损失函数
            loss = tf.reduce_mean(loss, axis=0)
            # 记录到tensorboard中
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)

            """
                保存模型
            """
            if (epoch + 1) % self.save_model_per_n_epoch == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        # 保存最终模型
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        end = time.time()
        print('Training Finished, Run Time: ', time.strftime('%H:%M:%S', time.gmtime(end - start)))







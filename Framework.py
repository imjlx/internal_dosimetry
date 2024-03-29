
import tensorflow as tf
import datetime
import time
import os
from tqdm import tqdm

from utils import Dataset
from utils import Loss, Metrics, Visual
from model import AutoEncoder

LOSS = Loss.l1_loss


class SequentialFramework(object):
    def __init__(self, model, train_IDs: tuple, test_IDs: tuple, batch: int):
        # 设置使用的网络模型
        self.model = model
        # 读取数据集
        self.train_ds = Dataset.create_dataset(train_IDs, batch)
        self.test_ds = Dataset.create_dataset(test_IDs, 1)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        # 检查点
        self.CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints\\" + self.CURRENT_TIME
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        # tensorboard
        train_log_dir = "logs\\" + self.CURRENT_TIME + "\\train"
        test_log_dir = "logs\\" + self.CURRENT_TIME + "\\test"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.save_model_per_n_epoch = 50

    @tf.function
    def train_step(self, ct, pet, source, dm):
        with tf.GradientTape() as g:
            dm_pred = self.model(inputs=[ct, pet, source], training=True)
            loss = LOSS(dm, dm_pred)
        gradient = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return loss

    @tf.function
    def test_step(self, ct, pet, source, dm):
        dm_pred = self.model(inputs=[ct, pet, source], training=False)
        loss = LOSS(dm, dm_pred)
        return loss

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
                bar.set_description("Epoch %i" % (epoch + 1))
                for n, (ct, pet, source, dm) in bar:
                    loss_add = self.train_step(ct, pet, source, dm)
                    loss.append(loss_add)

            loss = tf.reduce_mean(loss, axis=0)     # 对所有的批求平均，得到本轮的平均损失函数
            with self.train_summary_writer.as_default():    # 记录到tensorboard中
                tf.summary.scalar('loss', loss, step=epoch)

            """
            测试
            """
            loss = []
            for n, (ct, pet, source, dm) in bar:
                loss_add = self.test_step(ct, pet, source, dm)
                loss.append(loss_add)
            loss = tf.reduce_mean(loss, axis=0)
            with self.test_summary_writer.as_default():    # 记录到tensorboard中
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


class SequentialFrameworkSimple(object):
    def __init__(self, model: tf.keras.Model, train_IDs, test_IDs, batch_size=1):
        self.CURRENT_TIME = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        self.model = model
        self.train_ds = Dataset.create_dataset(patient_IDs=train_IDs, batch_size=batch_size)
        self.test_ds = Dataset.create_dataset(patient_IDs=test_IDs, batch_size=batch_size)
        self.batch_size = batch_size

    def Execute(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )
        # 检查点 与 tensorboard
        cp_path = "checkpoints\\" + self.CURRENT_TIME + "\\cp-{epoch:0>4d}.ckpd"
        tb_dir = "logs\\" + self.CURRENT_TIME

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'
        )
        self.model.save_weights(filepath=cp_path.format(epoch=0))

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)

        # 训练
        self.model.fit(
            x=self.train_ds, epochs=5,
            callbacks=[cp_callback, tb_callback],
            validation_data=self.test_ds, validation_freq=5
        )

        self.model.evaluate(x=self.test_ds)


if __name__ == "__main__":
    f = SequentialFrameworkSimple(AutoEncoder.AutoEncoder1(), train_IDs=(1, ), test_IDs=(2, ))
    f.Execute()
    pass


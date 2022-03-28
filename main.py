"""
def show_net_structure():
    net = AutoEncoder.AutoEncoder1()
    print(net.summary())
    tf.keras.utils.plot_model(net, to_file="AutoEncoder1.png", show_shapes=True, dpi=600)
"""

import Framework
from model import AutoEncoder


if __name__ == "__main__":
    f = Framework.SequentialFrameworkSimple(model=AutoEncoder.AutoEncoder1(), train_IDs=(1,), test_IDs=(2,))
    f.Execute()



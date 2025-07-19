import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_gru_arch():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()

    # 三个模块：Input、GRU、FC
    modules = [
        ("Input\n(50×41)", (0.05, 0.25, 0.25, 0.5)),
        ("GRU Layer\nhidden=128", (0.375, 0.25, 0.25, 0.5)),
        ("FC Layer\n128→14",      (0.7,  0.4,  0.25, 0.3))
    ]
    for label, (x, y, w, h) in modules:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center')

    # 箭头
    arrow_style = dict(arrowstyle="->", lw=2)
    ax.annotate("", xy=(0.30, 0.5), xytext=(0.375, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.625, 0.5), xytext=(0.7, 0.5),   arrowprops=arrow_style)

    plt.title('GRU Network Architecture')
    plt.show()

def plot_lstm_arch():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()

    # 三个模块：Input、LSTM、FC
    modules = [
        ("Input\n(50×41)", (0.05, 0.25, 0.25, 0.5)),
        ("LSTM Layer\nhidden=128\nlayers=2", (0.375, 0.25, 0.25, 0.5)),
        ("FC Layer\n128→14",      (0.7,  0.4,  0.25, 0.3))
    ]
    for label, (x, y, w, h) in modules:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center')

    # 箭头
    arrow_style = dict(arrowstyle="->", lw=2)
    ax.annotate("", xy=(0.30, 0.5), xytext=(0.375, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.625, 0.5), xytext=(0.7, 0.5),   arrowprops=arrow_style)

    plt.title('LSTM Network Architecture')
    plt.show()

def plot_gnn_arch():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axis_off()

    # 四个模块：Input、GAT、Transformer、FC
    modules = [
        ("Input\n(50×41)", (0.05, 0.25, 0.2, 0.5)),
        ("GAT Layer\nhidden=256\nheads=8", (0.3, 0.25, 0.2, 0.5)),
        ("Transformer\nlayers=3", (0.55, 0.25, 0.2, 0.5)),
        ("FC Layer\n2048→14", (0.8, 0.4, 0.2, 0.3))
    ]
    for label, (x, y, w, h) in modules:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center')

    # 箭头
    arrow_style = dict(arrowstyle="->", lw=2)
    ax.annotate("", xy=(0.25, 0.5), xytext=(0.3, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.5, 0.5), xytext=(0.55, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.75, 0.5), xytext=(0.8, 0.5), arrowprops=arrow_style)

    plt.title('GNN Network Architecture')
    plt.show()

if __name__ == "__main__":
    plot_gru_arch()
    plot_lstm_arch()
    plot_gnn_arch()

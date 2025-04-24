import matplotlib.pyplot as plt

def plot_fit(t, y_true, y_pred, label='Fit'):
    plt.plot(t, y_true, 'o', label='Experimental')
    plt.plot(t, y_pred, '-', label=label)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np


def plot_varianta(alpha,procent_minimal=80):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot varianta componente",
                 fontdict={"fontsize":16,"color":"b"})
    ax.set_xlabel("Componenta")
    ax.set_ylabel("Varianta")
    m = len(alpha)
    x = np.arange(1,m+1)
    ax.set_xticks(x)
    ax.plot(x,alpha)
    ax.scatter(x,alpha,c="r",alpha=0.5)
    procent_cumulat = np.cumsum( alpha*100/sum(alpha) )
    k1 = np.where(procent_cumulat>procent_minimal)[0][0]+1
    ax.axvline(k1,c="g",
               label="Varianta minimala (>"+str(procent_minimal)+"%)")
    k2 = np.where(alpha>1)[0][-1]+1
    ax.axvline(k2,c="m",label="Kaiser")
    ax.legend()
    plt.show()

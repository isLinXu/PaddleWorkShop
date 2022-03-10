import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 1D data
    x = np.linspace(0, 10, 100)
    y = np.cos(x)
    z = np.sin(x)

    # 2D data or images
    data = 2 * np.random.random((10, 10))
    data2 = 3 * np.random.random((10, 10))
    Y, X = np.mgrid[-3:3:100j, -3:3:100j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2

    from matplotlib.cbook import get_sample_data

    img = np.load(get_sample_data
                  ('axes_grid/bivariate_normal.npy'))

    # 1D data
    fig, ax = plt.subplots()
    lines = ax.plot(x, y)
    # Draw points with lines or makers connecting them
    ax.scatter(x, y)

    # figure
    fig1 = plt.figure()
    fig2 = plt.figure(figsize=plt.figaspect(2, 0))

    # axes
    fig.add_axes()
    ax1 = fig.add_subplot(221)
    # row-col-num
    ax3 = fig.add_subplot(212)
    fig3, axes = plt.subplots(
        nrows=2, ncols=2)
    fig4, axes2 = plt.subplots(ncols=3)

    # Draw unconnected points, scaled or colored
    axes[0, 0].bar([1, 2, 3], [3, 4, 5])
    # Plot vertical rectangles
    axes[1, 0].barh([0.5, 1, 2.5], [0, 1, 2])
    # Plot horizontal rectangles
    axes[1, 1].axhline(0.45)
    # Draw a horizontal line across axes
    axes[0, 1].axvline(0.65)
    # Draw a vertical line across axes
    ax.fill(x, y, color='blue')
    # Draw filled polygons
    ax.fill_between(x, y, color='yellow')
    # Fill between y-values and o

    # 2D data or images
    fig, ax = plt.subplots()
    # Colormapped or RGB arrays
    im = ax.imshow(img, cmap='gist_earth',
                   interpolation='nearest', vmin=-2, vmax=2)
    axes2[0].pcolor(data2)
    # Pseudocolor plot of 2D array
    axes2[0].pcolormesh(data)
    # Pseudocolor plot of 2D array
    CS = plt.contour(Y, X, U)
    # Plot contours
    axes2[2].contourf(data2)
    # Plot filled contours
    axes2[2] = ax.clabel(CS)
    # Label a contour plot

    # Vector Field
    axes[0, 1].arrow(0, 0, 0.5, 0.5)
    # Add an arrow to the axes
    axes[1, 1].quiver(y, z)
    # Plot a 2D field of arrows
    axes[0, 1].streamplot(X, Y, U, V)
    # plot a 2D field of arrows

    # Data Distributions
    ax1.hist(y)  # Plot a histogram
    ax3.boxplot(y)
    # make a box and whisker plot
    ax3.violinplot(z)
    # make a violin plot

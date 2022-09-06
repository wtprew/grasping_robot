import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=150)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None,
        save_folder='results',
        attempt=0,
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        fig.savefig(os.path.join(save_folder, '{}.png'.format(attempt)))

def plot_output(
        grasp_q_img=None,
        grasp_angle_img=None,
        grasp_width_img=None,
        save=False,
        save_folder='results',
        attempt=0,
        bins=3
    ):

    if bins > 1:
        fig1, axs = plt.subplots(3, 3, figsize=(15,15))
        ax1 = axs[0, 0]
        plot1 = ax1.imshow(grasp_q_img[0], cmap='Reds')
        ax1.set_title('Q')
        ax1.axis('off')
        fig1.colorbar(plot1, ax=ax1)
        ax4 = axs[1, 0]
        plot4 = ax4.imshow(grasp_q_img[1], cmap='Reds')
        ax4.axis('off')
        fig1.colorbar(plot4, ax=ax4)
        ax7 = axs[2, 0]
        plot7 = ax7.imshow(grasp_q_img[2], cmap='Reds')
        ax7.axis('off')
        fig1.colorbar(plot7, ax=ax7)
        ax2 = axs[0, 1]
        plot2 = ax2.imshow(grasp_angle_img[0], cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        ax2.set_title('Angle')
        ax2.axis('off')
        fig1.colorbar(plot2, ax=ax2, ticks=[-np.pi/2, 0, np.pi/2])
        ax5 = axs[1, 1]
        plot5 = ax5.imshow(grasp_angle_img[1], cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        ax5.axis('off')
        fig1.colorbar(plot5, ax=ax5, ticks=[-np.pi/2, 0, np.pi/2])
        ax8 = axs[2, 1]
        plot8 = ax8.imshow(grasp_angle_img[2], cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
        ax8.axis('off')
        fig1.colorbar(plot8, ax=ax8, ticks=[-np.pi/2, 0, np.pi/2])
        ax3 = axs[0, 2]
        plot3 = ax3.imshow(grasp_width_img[0], cmap='Reds')
        ax3.set_title('W')
        ax3.axis('off')
        fig1.colorbar(plot3, ax=ax3)
        ax6 = axs[1, 2]
        plot6 = ax6.imshow(grasp_width_img[1], cmap='Reds')
        ax6.axis('off')
        fig1.colorbar(plot6, ax=ax6)
        ax9 = axs[2, 2]
        plot9 = ax9.imshow(grasp_width_img[2], cmap='Reds')
        ax9.axis('off')
        fig1.colorbar(plot9, ax=ax9)
    
        plt.pause(0.1)
        fig1.canvas.draw()
    else:
        fig1, axs = plt.subplots(1, 3, figsize=(5,15))
        cmaps = ['Reds', 'RdBu', 'Reds']
        for col in range(3):
            ax = axs[col]
            if col == 0:
                output = ax.imshow(grasp_q_img, cmap=cmaps[col])
                ax.set_title('Q')
            elif col == 1:
                output = ax.imshow(grasp_angle_img, cmap=cmaps[col], vmin=-np.pi/2, vmax=np.pi/2)
                ax.set_title('Angle')
            else:
                output = ax.imshow(grasp_width_img, cmap=cmaps[col])
                ax.set_title('W')
            fig1.colorbar(output, ax=ax)
        plt.pause(0.1)
        fig1.canvas.draw()

    if save:
        # time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig1.savefig(os.path.join(save_folder, '{}_outputs.png'.format(attempt)))

def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    fig.savefig('results/rgb.png')

    if depth_img.any():
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        fig.savefig('results/depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')
    fig.savefig('results/grasp.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/quality.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/angle.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('results/width.png')

    fig.canvas.draw()
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import cv2
import sys

# Change this to the directory where you store KITTI data
BASEDIR= '/media/noneland/My Passport/KITTI/Object/training'
font = cv2.FONT_HERSHEY_SIMPLEX

def load_dataset(image_index, base_dir=BASEDIR):

    velo = np.fromfile(base_dir+"/velodyne/{:0>6d}.bin".format(image_index), dtype=np.float32) ## dtype must be set.
    print("Processing: {:0>6d}".format(image_index))
    velo = velo.reshape((-1, 4))

    image = cv2.imread(base_dir + '/image_2/{:0>6d}.png'.format(image_index))

    filename = base_dir + "/label_2/{:0>6d}.txt".format(image_index)
    with open(filename, 'r') as f:
        labels = [[eval(item) if index else item for index, item in enumerate(alabel.split())] for alabel in
                  f.readlines()]
    return image, velo, labels

def compute3DBox(labels):
    boxes3D = [ ]
    boxes2D = [ ]
    types = [ ]
    for label in labels:
        if label[0] in ['Car', 'Tram', 'Cyclist', 'Van', 'Truck', 'Pedestrian', 'Sitter']:
            types.append(label[0])
            rotation_y = label[14]
            R = np.array([[+math.cos(rotation_y), +math.sin(rotation_y), 0],
                         [ -math.sin(rotation_y), +math.cos(rotation_y), 0],
                         [                     0,                     0, 1]])
            h, w, l = tuple(label[8:11])

            tx = label[13]
            ty = -label[11]
            tz = -label[12]

            corners_3D = R.dot(np.array([
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]
                ])) + np.array([[tx], [ty], [tz]])
            boxes3D.append(corners_3D)
            boxes2D.append(label[4:8])
    return boxes2D, boxes3D, types


colors2D = {'Car': (255, 0, 0),
    'Truck': (255, 0, 0),
    'Van': (255, 0, 0),
    'Sitter': (0, 255, 0),
    'Tram': (255, 0, 0),
    'Cyclist': (0, 255, 0),
    'Pedestrian': (0, 0, 255),
}

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
axes_limits = [
    [-80, 80],  # X axis range
    [-60, 60],  # Y axis range
    [-8, 3]  # Z axis range
]
axes_str = ['X', 'Y', 'Z']


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def display_frame_statistics(dataset, image, boxes2d, tracklet_rects=None, tracklet_types=None,  points=0.2):
    """
    Displays statistics for a single frame. Draws camera data, 3D plot of the lidar point cloud data and point cloud
    projections to various planes.

    Parameters
    ----------
    dataset         : `raw` dataset.
    tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
    tracklet_types  : Dictionary with tracklet types.
    frame           : Absolute number of the frame.
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
    """
    # dataset_rgb = list(dataset.rgb)
    # f, ax = plt.subplots(1, 1, figsize=(15, 5))
    # ax[0, 0].imshow(dataset_rgb[frame][0], cmap='gray')
    # ax[0, 0].set_title('RGB Image (cam2)')
    # plt.show()
    # print(dataset.shape)
    # print("xlim:", np.max(dataset[:, 0]), np.min(dataset[:, 0]))
    # print("ylim:", np.max(dataset[:, 1]), np.min(dataset[:, 1]))
    # print("zlim:", np.max(dataset[:, 2]), np.min(dataset[:, 2]))
    # print("rlim:", np.max(dataset[:, 3]), np.min(dataset[:, 3]))
    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset.shape[0], points_step)
    velo_frame = dataset[velo_range, :]

    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """

        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d != None:
            ax.set_xlim3d(xlim3d)
        if ylim3d != None:
            ax.set_ylim3d(ylim3d)
        if zlim3d != None:
            ax.set_zlim3d(zlim3d)

        for t_rects, t_type in zip(tracklet_rects, tracklet_types):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])
            # Draw point cloud data as plane projections
    f3, ax3 = plt.subplots(3, 1, figsize=(15, 25), num=0)
    draw_point_cloud(
        ax3[0],
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right',
        axes=[0, 2]  # X and Z axes
    )
    draw_point_cloud(
        ax3[1],
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right',
        axes=[0, 1]  # X and Y axes
    )
    draw_point_cloud(
        ax3[2],
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane',
        axes=[1, 2]  # Y and Z axes
    )
    f3.show()

    # Draw point cloud data as 3D plot
    f2 = plt.figure(1, figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-80, 80))
    f2.show()



    for i in range(len(boxes2D)):
        box = [int(coor) for coor in boxes2D[i]]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors2D[types[i]])
        cv2.putText(image, types[i], (box[0], box[1]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Cam2", image)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give the image number.")
        sys.exit(-1)
    try:
        frame = eval(sys.argv[1])
        print(frame)
        if isinstance(frame, int) and 0 <= frame <= 7481:
            print("frame num reasonnale.")
        else:
            print("Parameter error. Please Check!")
            sys.exit(1)
    except:
        print("Can't eval parameter. Please Check!")
        sys.exit(1)
    while True:
        image, velo, labels = load_dataset(frame)
        boxes2D, boxes3D, types = compute3DBox(labels)
        display_frame_statistics(velo, image, boxes2D, boxes3D, types)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key == ord('n'):
            frame += 1
            plt.close("all")
__author__  = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

# GUI adapted from
# https://memotut.com/create-a-3d-model-viewer-with-pyqt5-and-pyqtgraph-b3916/ and
# https://matplotlib.org/3.1.1/gallery/user_interfaces/embedding_in_qt_sgskip.html

import os
import copy
import gzip
import socket
import pandas
import logging
import argparse

import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.cm as cm
import multiprocessing as mp
import pyqtgraph.opengl as gl

from glob import glob
from typing import List
from pathlib import Path
from plyfile import PlyData

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui
from fog_simulation import ParameterSet, RNG, simulate_fog

from SeeingThroughFog.tools.DatasetViewer.dataset_viewer import load_calib_data, read_label
from SeeingThroughFog.tools.DatasetFoggification.beta_modification import BetaRadomization
from SeeingThroughFog.tools.DatasetFoggification.lidar_foggification import haze_point_cloud

          #  R,     G,   B,  alpha
COLORS = [(  0,   255,   0,   255),  # cars in green
          (255,     0,   0,   255),  # pedestrian in red
          (255,   255,   0,   255)]  # cyclists in yellow


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, help='path to where you store your datasets',
                    default=str(Path.home() / 'datasets'))
parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                    default=str(Path.home() / 'repositories/PCDet/output'))
args = parser.parse_args()

DATASETS_ROOT = Path(args.datasets)
EXPERIMENTS_ROOT = Path(args.experiments)

  FOG    = DATASETS_ROOT/'DENSE/SeeingThroughFog/lidar_hdl64_strongest_fog_extraction'
  AUDI   = DATASETS_ROOT/'A2D2/camera_lidar_semantic_bboxes'
  LYFT   = DATASETS_ROOT/'LyftLevel5/Perception/train_lidar'
  ARGO   = DATASETS_ROOT/'Argoverse'
  PANDA  = DATASETS_ROOT/'PandaSet'
  DENSE  = DATASETS_ROOT/'DENSE/SeeingThroughFog/lidar_hdl64_strongest'
  KITTI  = DATASETS_ROOT/'KITTI/3D/training/velodyne'
  WAYMO  = DATASETS_ROOT/'WaymoOpenDataset/WOD/train/velodyne'
  HONDA  = DATASETS_ROOT/'Honda_3D/scenarios'
  APOLLO = DATASETS_ROOT/'Apollo3D'
NUSCENES = DATASETS_ROOT/'nuScenes/sweeps/LIDAR_TOP'

if socket.gethostname() == 'beast':
    DENSE = Path.home() / 'datasets_local' / 'DENSE/SeeingThroughFog/lidar_hdl64_strongest'

def get_extracted_fog_file_list(dirname: str) -> List[str]:
    file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.bin'))]
    return sorted(file_list)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MyWindow(QMainWindow):
    def __init__(self) -> None:
        super(MyWindow, self).__init__()
        self.boxes = {}
        self.predictions = {}
        self.result_dict = {}
        self.show_predictions = True
        self.prediction_threshold = 50

        self.gain = True
        self.noise_variant = 'v4'

        self.noise = 10
        self.noise_min = 0
        self.noise_max = 20

        self.min_fog_response = -1
        self.max_fog_response = -1
        self.num_fog_responses = -1

        self.num_cpus = mp.cpu_count()
        self.pool = mp.Pool(self.num_cpus)

        self.p = ParameterSet(gamma=0.000001,
                              gamma_min=0.0000001,
                              gamma_max=0.00001,
                              gamma_scale=10000000)

        self.p.beta_0 = self.p.gamma / np.pi
        self.row_height = 20

        hostname = socket.gethostname()

        if hostname == 'beast':
            self.monitor = QDesktopWidget().screenGeometry(1)
            self.monitor.setHeight(int(0.45 * self.monitor.height()))
        elif hostname == 'hox':
            self.monitor = QDesktopWidget().screenGeometry(2)
            self.monitor.setHeight(int(0.45 * self.monitor.height()))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(self.monitor.height())

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)
        self.simulated_fog = False
        self.simulated_fog_pc = None

        self.simulated_fog_dense = False
        self.extracted_fog = False
        self.extracted_fog_pc = None
        self.extracted_fog_index = -1
        self.extracted_fog_mesh = None
        self.extracted_fog_file_list = None

        self.color_dict = {0: 'x',
                           1: 'y',
                           2: 'z',
                           3: 'intensity',
                           4: 'distance',
                           5: 'angle',
                           6: 'channel'}

        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.color_feature = 2
        self.point_size = 3
        self.threshold = 50
        self.dataset = None
        self.success = False
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_name = self.color_dict[self.color_feature]

        self.lastDir = None
        self.current_pc = None
        self.fogless_pc = None
        self.current_mesh = None
        self.droppedFilename = None

        self.file_name = None
        self.file_list = None
        self.index = -1

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)

        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        self.grid_dimensions = 20
        self.viewer = gl.GLViewWidget()
        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 6)

        self.grid = gl.GLGridItem()
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)

        self.reset_btn = QPushButton("reset")
        self.reset_btn.clicked.connect(self.reset)
        self.layout.addWidget(self.reset_btn, 1, 3, 1, 2)

        self.load_kitti_btn = QPushButton("KITTI")
        self.load_kitti_btn.clicked.connect(self.load_kitti)
        self.layout.addWidget(self.load_kitti_btn, 2, 3)

        self.load_dense_btn = QPushButton("DENSE")
        self.load_dense_btn.clicked.connect(self.load_dense)
        self.layout.addWidget(self.load_dense_btn, 3, 3)

        self.load_honda_btn = QPushButton("H3D")
        self.load_honda_btn.clicked.connect(self.load_honda)
        self.layout.addWidget(self.load_honda_btn, 4, 3)

        self.load_audi_btn = QPushButton("A2D2")
        self.load_audi_btn.clicked.connect(self.load_audi)
        self.layout.addWidget(self.load_audi_btn, 5, 3)

        self.load_panda_btn = QPushButton("PandaSet")
        self.load_panda_btn.clicked.connect(self.load_panda)
        self.layout.addWidget(self.load_panda_btn, 6, 3)

        self.load_nuscenes_btn = QPushButton("nuScenes")
        self.load_nuscenes_btn.clicked.connect(self.load_nuscenes)
        self.layout.addWidget(self.load_nuscenes_btn, 2, 4)

        self.load_lyft_btn = QPushButton("LyftL5")
        self.load_lyft_btn.clicked.connect(self.load_lyft)
        self.layout.addWidget(self.load_lyft_btn, 3, 4)

        self.load_kitti_btn = QPushButton("Argoverse")
        self.load_kitti_btn.clicked.connect(self.load_argo)
        self.layout.addWidget(self.load_kitti_btn, 4, 4)

        self.load_waymo_btn = QPushButton("Waymo")
        self.load_waymo_btn.clicked.connect(self.load_waymo)
        self.layout.addWidget(self.load_waymo_btn, 5, 4)

        self.load_apollo_btn = QPushButton("Apollo")
        self.load_apollo_btn.clicked.connect(self.load_apollo)
        self.layout.addWidget(self.load_apollo_btn, 6, 4)

        self.load_fog_btn = QPushButton("extracted fog samples")
        self.load_fog_btn.clicked.connect(self.load_extracted_fog_samples)
        self.layout.addWidget(self.load_fog_btn, 7, 3, 1, 2)

        if self.extracted_fog:
            self.toggle_extracted_fog_btn = QPushButton("remove extracted fog")
        else:
            self.toggle_extracted_fog_btn = QPushButton("add extracted fog")

        self.toggle_extracted_fog_btn.clicked.connect(self.toggle_extracted_fog)
        self.layout.addWidget(self.toggle_extracted_fog_btn, 8, 3, 1, 2)

        if self.simulated_fog:
            self.toggle_simulated_fog_btn = QPushButton("remove our fog simulation")
        else:
            self.toggle_simulated_fog_btn = QPushButton("add our fog simulation")

        self.toggle_simulated_fog_btn.clicked.connect(self.toggle_simulated_fog)
        self.layout.addWidget(self.toggle_simulated_fog_btn, 9, 3, 1, 2)

        self.choose_dir_btn = QPushButton("choose custom directory")
        self.choose_dir_btn.clicked.connect(self.show_directory_dialog)
        self.layout.addWidget(self.choose_dir_btn, 2, 1)

        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")

        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

        self.layout.addWidget(self.prev_btn, 2, 0)
        self.layout.addWidget(self.next_btn, 2, 2)

        self.color_title = QLabel("color code")
        self.color_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_title, 3, 0)

        self.color_label = QLabel(self.color_name)
        self.color_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_label, 3, 2)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(0)
        self.color_slider.setMaximum(6)
        self.color_slider.setValue(self.color_feature)
        self.color_slider.setTickPosition(QSlider.TicksBelow)
        self.color_slider.setTickInterval(1)

        self.layout.addWidget(self.color_slider, 3, 1)
        self.color_slider.valueChanged.connect(self.color_slider_change)

        self.threshold_title = QLabel("fog threshold")
        self.threshold_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.threshold_title, 4, 0)

        self.threshold_label = QLabel(str(self.threshold))
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.threshold_label, 4, 2)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(1)

        self.layout.addWidget(self.threshold_slider, 4, 1)
        self.threshold_slider.valueChanged.connect(self.threshold_slider_change)

        self.file_name_label = QLabel()
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.file_name_label.setMaximumSize(self.monitor.width(), 20)
        self.layout.addWidget(self.file_name_label, 1, 1, 1, 1)

        self.reset_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.toggle_extracted_fog_btn.setEnabled(False)
        self.toggle_simulated_fog_btn.setEnabled(False)

        ###########################
        # fog simulation controls #
        ###########################

        self.current_row = 5

        self.mor_label = QLabel(f'meteorological optical range (MOR) = {round(self.p.mor, 2)}m')
        self.mor_label.setAlignment(Qt.AlignCenter)
        self.mor_label.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.mor_label, self.current_row, 1)
        self.current_row += 1

        self.alpha_title = QLabel('attenuation coefficient')
        self.alpha_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.alpha_title, self.current_row, 0)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(int(self.p.alpha_min * self.p.alpha_scale))
        self.alpha_slider.setMaximum(int(self.p.alpha_max * self.p.alpha_scale))
        self.alpha_slider.setValue(int(self.p.alpha * self.p.alpha_scale))

        self.layout.addWidget(self.alpha_slider, self.current_row, 1)
        self.alpha_slider.valueChanged.connect(self.update_labels)

        self.alpha_label = QLabel(f"\u03B1 = {self.p.alpha}")
        self.alpha_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.alpha_label, self.current_row, 2)
        self.current_row += 1

        self.beta_title = QLabel('backscattering coefficient')
        self.beta_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.beta_title, self.current_row, 0)

        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setMinimum(int(self.p.beta_min * self.p.beta_scale))
        self.beta_slider.setMaximum(int(self.p.beta_max * self.p.beta_scale))
        self.beta_slider.setValue(int(self.p.beta * self.p.beta_scale))

        self.layout.addWidget(self.beta_slider, self.current_row, 1)
        self.beta_slider.valueChanged.connect(self.update_labels)

        self.beta_label = QLabel(f"\u03B2 = {round(self.p.beta * self.p.mor, 3)} / MOR")
        self.beta_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.beta_label, self.current_row, 2)
        self.current_row += 1

        self.gamma_title = QLabel("reflextivity of the hard target")
        self.gamma_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.gamma_title, self.current_row, 0)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(int(self.p.gamma_min * self.p.gamma_scale))
        self.gamma_slider.setMaximum(int(self.p.gamma_max * self.p.gamma_scale))
        self.gamma_slider.setValue(int(self.p.gamma * self.p.gamma_scale))

        self.layout.addWidget(self.gamma_slider, self.current_row, 1)
        self.gamma_slider.valueChanged.connect(self.update_labels)

        self.gamma_label = QLabel(f"\u0393 = {self.p.gamma}")
        self.gamma_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.gamma_label, self.current_row, 2)
        self.current_row += 1

        self.noise_title = QLabel("spread of relative noise")
        self.noise_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.noise_title, self.current_row, 0)

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(self.noise_min)
        self.noise_slider.setMaximum(self.noise_max)
        self.noise_slider.setValue(self.noise)

        self.layout.addWidget(self.noise_slider, self.current_row, 1)
        self.noise_slider.valueChanged.connect(self.update_labels)

        self.noise_label = QLabel(f"{self.noise}m")
        self.noise_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.noise_label, self.current_row, 2)
        self.current_row += 1

        self.num_info = QLabel("")
        self.num_info.setAlignment(Qt.AlignLeft)
        self.num_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.num_info, self.current_row, 0)

        self.log_info = QLabel("")
        self.log_info.setAlignment(Qt.AlignLeft)
        self.log_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.log_info, self.current_row, 1, 1, 3)

        if self.extracted_fog:
            self.threshold_slider.setEnabled(True)
        else:
            self.threshold_slider.setEnabled(False)

        if self.simulated_fog:
            self.toggle_simulated_fog_btn.setText('remove our fog simulation')
            self.alpha_slider.setEnabled(True)
            self.beta_slider.setEnabled(True)
            self.gamma_slider.setEnabled(True)
            # self.noise_slider.setEnabled(True)
        else:
            self.toggle_simulated_fog_btn.setText('add our fog simulation')
            self.alpha_slider.setEnabled(False)
            self.beta_slider.setEnabled(False)
            self.gamma_slider.setEnabled(False)
            # self.noise_slider.setEnabled(False)

        self.dense_split_paths = []

        self.cb = QComboBox()
        # self.cb.setEditable(True)
        # self.cb.lineEdit().setReadOnly(True)
        # self.cb.lineEdit().setAlignment(Qt.AlignCenter)
        self.cb.addItems(self.populate_dense_splits())
        self.cb.currentIndexChanged.connect(self.selection_change)
        self.cb.setEnabled(False)
        self.layout.addWidget(self.cb, self.current_row, 3, 1, 2)
        self.current_row += 1

        if self.simulated_fog_dense:
            self.toggle_simulated_fog_dense_btn = QPushButton("remove STF fog simulation")
        else:
            self.toggle_simulated_fog_dense_btn = QPushButton("add STF fog simulation")

        self.toggle_simulated_fog_dense_btn.clicked.connect(self.toggle_simulated_fog_dense)
        self.layout.addWidget(self.toggle_simulated_fog_dense_btn, self.current_row, 3, 1, 2)
        self.toggle_simulated_fog_dense_btn.setEnabled(False)

        # Create textbox
        if self.show_predictions:
            self.visualize_predictions_path_btn = QPushButton('hide predictions', self)
        else:
            self.visualize_predictions_path_btn = QPushButton('show predictions', self)
        self.visualize_predictions_path_btn.setEnabled(False)
        self.layout.addWidget(self.visualize_predictions_path_btn, self.current_row, 0, 1, 1)
        self.visualize_predictions_path_btn.clicked.connect(self.toggle_predictions)
        self.experiment_path_box = QLineEdit(self)
        self.experiment_path_box.setText('dense_models/pv_rcnn/'
                                         '2021-03-13_01-29-41_98d97b57_dense_models_batch_size_24_pv_rcnn')
        self.layout.addWidget(self.experiment_path_box, self.current_row, 1, 1, 1)
        self.load_experiment_path_btn = QPushButton('load results', self)
        self.layout.addWidget(self.load_experiment_path_btn, self.current_row, 2, 1, 1)
        self.load_experiment_path_btn.clicked.connect(self.load_results)

        self.current_row += 1

        self.prediction_threshold_title = QLabel("prediction confidence")
        self.prediction_threshold_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_threshold_title, self.current_row, 0)

        self.prediction_threshold_label = QLabel(str(self.prediction_threshold))
        self.prediction_threshold_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_threshold_label, self.current_row, 2)

        self.prediction_threshold_slider = QSlider(Qt.Horizontal)
        self.prediction_threshold_slider.setMinimum(0)
        self.prediction_threshold_slider.setMaximum(100)
        self.prediction_threshold_slider.setValue(self.prediction_threshold)
        self.prediction_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.prediction_threshold_slider.setTickInterval(10)
        self.prediction_threshold_slider.setEnabled(False)

        self.layout.addWidget(self.prediction_threshold_slider, self.current_row, 1)
        self.prediction_threshold_slider.valueChanged.connect(self.prediction_threshold_slider_change)

        self.current_row += 1

        # hide broken functionality for now
        self.toggle_extracted_fog_btn.setVisible(False)
        self.load_fog_btn.setVisible(False)

        self.threshold_title.setVisible(False)
        self.threshold_slider.setVisible(False)
        self.threshold_label.setVisible(False)

        self.noise_title.setVisible(False)
        self.noise_slider.setVisible(False)
        self.noise_label.setVisible(False)


    def load_results(self) -> None:
        exp_dir = EXPERIMENTS_ROOT / self.experiment_path_box.text()
        test_folders = [x[0] for x in os.walk(exp_dir) if 'epoch' in x[0] and 'test' in x[0]]
        self.result_dict = {}
        for test_folder in test_folders:
            key = test_folder.split('/')[-1]
            pkl_path = Path(test_folder) / 'result.pkl'
            with open(pkl_path, 'rb') as f:
                self.result_dict[key] = pkl.load(f)
        if self.result_dict is not None:
            self.visualize_predictions_path_btn.setEnabled(True)
            self.prediction_threshold_slider.setEnabled(True)

    def visualize_predictions(self) -> None:
        split = self.cb.currentText()
        if 'test' in split:
            pred_dict = self.result_dict[split][self.index]
            assert self.file_name.split('/')[-1].split('.')[0] == pred_dict['frame_id'], f'frame missmatch ' \
                f"{self.file_name.split('/')[-1].split('.')[0]} != {pred_dict['frame_id']}"
            lookup = {'Car': 0,
                      'Pedestrian': 1,
                      'Cyclist': 2}
            predictions = np.zeros((pred_dict['boxes_lidar'].shape[0], 9))
            predictions[:, 0:-2] = pred_dict['boxes_lidar']
            predictions[:, 7] = np.array([lookup[name] for name in pred_dict['name']])
            predictions[:, 8] = pred_dict['score']

            for prediction in predictions:
                x, y, z, w, l, h, rotation, category, score = prediction
                if score*100 > self.prediction_threshold:
                    dist = np.sqrt(x**2 + y**2 + z**2)
                    rotation = np.rad2deg(rotation) + 90
                    color = (255, 255, 255, 255)        # white
                    box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)
                    box.setSize(l, w, h)
                    box.translate(-l / 2, -w / 2, -h / 2)
                    box.rotate(angle=rotation, x=0, y=0, z=1)
                    box.translate(x, y, z)
                    self.viewer.addItem(box)
                    self.predictions[dist] = box

    def populate_dense_splits(self) -> List[str]:
        split_folder = Path(__file__).parent.absolute() / 'SeeingThroughFog' / 'splits'
        splits = []
        for file in os.listdir(split_folder):
            if file.endswith('.txt'):
                splits.append(file.replace('.txt', ''))
                self.dense_split_paths.append(split_folder / file)
        self.dense_split_paths = sorted(self.dense_split_paths)
        return sorted(splits)

    def selection_change(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        split = self.cb.currentText()
        # open file and read the content in a list
        with open(f'SeeingThroughFog/splits/{split}.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(DENSE) / f'{line[:-1].replace(",", "_")}.bin'

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_dense()
        self.show_pointcloud(self.file_list[self.index])


    def update_labels(self) -> None:
        self.p.alpha = self.alpha_slider.value() / self.p.alpha_scale
        self.alpha_label.setText(f"\u03B1 = {self.p.alpha}")

        self.p.mor = np.log(20) / self.p.alpha
        self.mor_label.setText(f'meteorological optical range (MOR) = {round(self.p.mor, 2)}m')

        self.p.beta_scale = 1000 * self.p.mor
        self.p.beta = self.beta_slider.value() / self.p.beta_scale
        self.beta_label.setText(f"\u03B2 = {round(self.p.beta * self.p.mor, 3)} / MOR")

        self.p.gamma = self.gamma_slider.value() / self.p.gamma_scale
        self.gamma_label.setText(f"\u0393 = {self.p.gamma}")
        self.p.beta_0 = self.p.gamma / np.pi

        self.noise = self.noise_slider.value()
        self.noise_label.setText(f"{self.noise}m")

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])


    def reset_fog_buttons(self) -> None:
        self.boxes = {}
        self.threshold_slider.setEnabled(False)
        self.alpha_slider.setEnabled(False)
        self.beta_slider.setEnabled(False)
        self.gamma_slider.setEnabled(False)
        self.noise_slider.setEnabled(False)

        self.simulated_fog = False
        self.simulated_fog_dense = False
        self.toggle_simulated_fog_btn.setText('add our fog simulation')
        self.toggle_simulated_fog_dense_btn.setText('add STF fog simulation')

        self.extracted_fog = False
        self.toggle_extracted_fog_btn.setText('add extraced fog')


    def reset(self) -> None:
        self.reset_viewer()
        self.reset_fog_buttons()
        self.reset_custom_values()

        self.cb.setEnabled(False)
        self.toggle_extracted_fog_btn.setEnabled(False)
        self.toggle_simulated_fog_btn.setEnabled(False)
        self.toggle_simulated_fog_dense_btn.setEnabled(False)

        self.file_list = None

        self.current_pc = None
        self.current_mesh = None

        self.simulated_fog = False
        self.simulated_fog_dense = False

        self.extracted_fog = False
        self.extracted_fog_pc = None
        self.extracted_fog_mesh = None
        self.extracted_fog_index = -1

    def reset_custom_values(self) -> None:
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.dataset = None
        self.success = False
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_name = self.color_dict[self.color_feature]

    def threshold_slider_change(self) -> None:
        self.threshold = self.threshold_slider.value()
        self.threshold_label.setText(str(self.threshold))
        if self.current_mesh and Path(self.file_name).suffix != '.pickle':
            self.show_pointcloud(self.file_name)


    def prediction_threshold_slider_change(self) -> None:
        self.prediction_threshold = self.prediction_threshold_slider.value()
        self.prediction_threshold_label.setText(str(self.prediction_threshold))
        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def color_slider_change(self) -> None:
        self.color_feature = self.color_slider.value()
        self.color_name = self.color_dict[self.color_feature]
        self.color_label.setText(self.color_name)

        if self.current_mesh:
            if Path(self.file_name).suffix == '.pickle':
                self.show_pcdet_dict(self.file_name)
            else:
                self.show_pointcloud(self.file_name)


    def check_index_overflow(self) -> None:
        if self.index == -1:
            self.index = len(self.file_list) - 1

        if self.index >= len(self.file_list):
            self.index = 0


    def decrement_index(self) -> None:
        if self.index != -1:
            self.index -= 1
            self.check_index_overflow()
            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])


    def increment_index(self) -> None:
        if self.index != -1:
            self.index += 1
            self.check_index_overflow()

            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])


    def set_pc_det(self, before: bool) -> None:

        self.min_value = 0
        self.max_value = 63
        self.extension = 'pickle'
        self.d_type = np.float32
        self.intensity_multiplier = 1

        if before:
            self.dataset = 'before'
            self.num_features = 5
            self.color_dict[6] = 'channel'
        else:
            self.dataset = 'after'
            self.num_features = 4
            self.color_dict[6] = 'not available'


    def set_extracted_fog_samples(self):
        self.threshold_slider.setEnabled(True)
        self.dataset = 'FOG'
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def set_kitti(self) -> None:
        self.dataset = 'KITTI'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'

    def load_kitti(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/KITTI.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(KITTI) / line[:-1]
                # add item to the list
                self.file_list.append(str(file_path))
        self.index = 0
        self.set_kitti()
        self.show_pointcloud(self.file_list[self.index])


    def set_audi(self) -> None:
        self.dataset = 'A2D2'
        self.min_value = 0
        self.max_value = 4
        self.num_features = 5
        self.extension = 'npz'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'lidar_id'

    def load_audi(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/A2D2.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(AUDI) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_audi()
        self.show_pointcloud(self.file_list[self.index])


    def set_honda(self) -> None:
        self.dataset = 'Honda3D'
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.extension = 'ply'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_honda(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/Honda3D.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(HONDA) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))
        self.index = 0
        self.set_honda()
        self.show_pointcloud(self.file_list[self.index])


    def set_argo(self) -> None:
        self.dataset = 'Argoverse'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.extension = 'ply'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_argo(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/Argoverse.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(ARGO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_argo()
        self.show_pointcloud(self.file_list[self.index])


    def set_dense(self) -> None:
        self.dataset = 'DENSE'
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

        self.cb.setEnabled(True)


    def load_dense(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/DENSE.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(DENSE) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_dense()
        self.cb.setCurrentText('all')
        self.show_pointcloud(self.file_list[self.index])


    def set_nuscenes(self) -> None:
        self.dataset = 'nuScenes'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'


    def load_nuscenes(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []

        with open('file_lists/nuScenes.pkl', 'rb') as f:
            file_list = pkl.load(f)

        for file in file_list:
            self.file_list.append(NUSCENES + file)

        self.index = 0
        self.set_nuscenes()
        self.show_pointcloud(self.file_list[self.index])


    def set_lyft(self) -> None:
        self.dataset = 'LyftL5'
        self.min_value = 0
        self.max_value = 16
        self.num_features = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'


    def load_lyft(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []

        # open file and read the content in a list
        with open('file_lists/LyftL5.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(LYFT) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_lyft()
        self.show_pointcloud(self.file_list[self.index])


    def set_waymo(self) -> None:
        self.dataset = 'WaymoOpenDataset'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'


    def load_waymo(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/WAYMO.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(WAYMO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_waymo()
        self.show_pointcloud(self.file_list[self.index])


    def set_panda(self) -> None:
        self.dataset = 'PandaSet'
        self.min_value = 0
        self.max_value = 1
        self.num_features = 5
        self.extension = 'pkl.gz'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'lidar_id'


    def load_panda(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/PandaSet.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(PANDA) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_panda()
        self.show_pointcloud(self.file_list[self.index])


    def set_apollo(self) -> None:
        self.dataset = 'Apollo'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'


    def load_apollo(self) -> None:
        self.reset_fog_buttons()
        self.file_list = []
        # open file and read the content in a list
        with open('file_lists/Apollo.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(APOLLO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_apollo()
        self.show_pointcloud(self.file_list[self.index])


    def show_directory_dialog(self) -> None:
        self.reset_fog_buttons()
        directory = Path(os.getenv("HOME")) / 'Downloads'
        if self.lastDir:
            directory = self.lastDir
        dir_name = QFileDialog.getExistingDirectory(self, "Open Directory", str(directory),
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        if dir_name:
            self.create_file_list(dir_name)
            self.lastDir = Path(dir_name)


    def get_index(self, filename: str) -> int:
        try:
            return self.file_list.index(str(filename))

        except ValueError:
            logging.warning(f'{filename} not found in self.file_list')
            return -1


    def load_extracted_fog_samples(self):
        self.set_extracted_fog_samples()
        self.create_file_list(FOG)


    def toggle_simulated_fog(self) -> None:
        if self.file_list is not None and 'extraction' not in self.file_name:
            self.simulated_fog = not self.simulated_fog
            if self.simulated_fog:
                self.toggle_simulated_fog_btn.setText('remove our fog simulation')
                self.alpha_slider.setEnabled(True)
                self.beta_slider.setEnabled(True)
                self.gamma_slider.setEnabled(True)
                self.noise_slider.setEnabled(True)
            else:
                self.toggle_simulated_fog_btn.setText('add our fog simulation')
                self.alpha_slider.setEnabled(False)
                self.beta_slider.setEnabled(False)
                self.gamma_slider.setEnabled(False)
                self.noise_slider.setEnabled(False)
            self.show_pointcloud(self.file_list[self.index])

    def toggle_simulated_fog_dense(self) -> None:
        if self.file_list is not None and 'extraction' not in self.file_name:
            self.simulated_fog_dense = not self.simulated_fog_dense
            if self.simulated_fog_dense:
                self.toggle_simulated_fog_dense_btn.setText('remove STF fog simulation')
                self.alpha_slider.setEnabled(True)
                self.beta_slider.setEnabled(True)
                self.gamma_slider.setEnabled(True)
                self.noise_slider.setEnabled(True)
            else:
                self.toggle_simulated_fog_dense_btn.setText('add STF fog simulation')
                self.alpha_slider.setEnabled(False)
                self.beta_slider.setEnabled(False)
                self.gamma_slider.setEnabled(False)
                self.noise_slider.setEnabled(False)
            self.show_pointcloud(self.file_list[self.index])

    def toggle_predictions(self) -> None:
        self.show_predictions = not self.show_predictions
        if self.show_predictions:
            self.visualize_predictions_path_btn.setText('hide predictions')
        else:
            self.visualize_predictions_path_btn.setText('show predictions')
        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])


    def toggle_extracted_fog(self) -> None:
        if self.file_list is not None and 'extraction' not in self.file_name:
            if self.extracted_fog_file_list is None:
                self.extracted_fog_file_list = get_extracted_fog_file_list(FOG)
            self.extracted_fog = not self.extracted_fog
            if self.extracted_fog:
                self.threshold_slider.setEnabled(True)
                self.toggle_extracted_fog_btn.setText('remove extracted fog')
            else:
                self.threshold_slider.setEnabled(False)
                self.toggle_extracted_fog_btn.setText('add extracted fog')
                self.extracted_fog_index = -1
            self.show_pointcloud(self.file_list[self.index])

            # TODO: workaround because the slider has no effect yet
            if self.extension == 'pickle':
                self.threshold_slider.setEnabled(False)

    def create_file_list(self, dirname: str, filename: str = None, extension: str =None) -> None:
        if extension:
            file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.{extension}'))]
        else:
            file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.{self.extension}'))]

        self.file_list = sorted(file_list)

        # with open('file_lists/Argoverse.txt', 'w') as filehandle:
        #     filehandle.writelines(f'{Path(file).parent.parent.parent.parent.name}/'
        #                           f'{Path(file).parent.parent.parent.name}/'
        #                           f'{Path(file).parent.parent.name}/'
        #                           f'{Path(file).parent.name}/'
        #                           f'{Path(file).name}\n' for file in self.file_list)

        if len(self.file_list) > 0:
            if filename is None:
                filename = self.file_list[0]
            self.index = self.get_index(filename)
            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])


    def reset_viewer(self) -> None:
        if self.file_name:
            if 'extraction' in self.file_name or self.extracted_fog:
                self.threshold_slider.setEnabled(True)
            else:
                self.threshold_slider.setEnabled(False)

        self.num_info.setText(f'sequence_size: {len(self.file_list)}')

        self.min_fog_response = np.inf
        self.max_fog_response = 0
        self.num_fog_responses = 0

        self.viewer.items = []
        self.viewer.addItem(self.grid)


    def show_pcdet_dict(self, filename: str) -> None:
        self.reset_viewer()
        self.simulated_fog = False
        self.simulated_fog_dense = False
        if self.simulated_fog:
            self.toggle_simulated_fog_btn.setText('remove our fog simulation')
            self.alpha_slider.setEnabled(True)
            self.beta_slider.setEnabled(True)
            self.gamma_slider.setEnabled(True)
            self.noise_slider.setEnabled(True)
        else:
            self.toggle_simulated_fog_btn.setText('add our fog simulation')
            self.alpha_slider.setEnabled(False)
            self.beta_slider.setEnabled(False)
            self.gamma_slider.setEnabled(False)
            self.noise_slider.setEnabled(False)
        self.file_name = filename
        self.set_pc_det('before' in filename)
        self.cb.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.toggle_extracted_fog_btn.setEnabled(True)
        self.toggle_simulated_fog_btn.setEnabled(True)
        self.toggle_simulated_fog_dense_btn.setEnabled(True)

        if len(self.file_list) > 1:
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
        else:
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)

        pcdet_dict = pkl.load(open(filename, "rb"))

        ##########
        # points #
        ##########

        pc = pcdet_dict['points']
        self.log_string(pc)
        colors = self.get_colors(pc)
        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
        self.current_mesh = mesh
        self.current_pc = copy.deepcopy(pc)
        self.fogless_pc = copy.deepcopy(pc)
        self.viewer.addItem(mesh)

        #########
        # boxes #
        #########
        self.boxes = {}
        self.create_boxes(pcdet_dict['gt_boxes'])


    def create_boxes(self, annotations):
        # create annotation boxes
        for annotation in annotations:
            x, y, z, w, l, h, rotation, category = annotation
            rotation = np.rad2deg(rotation) + 90
            try:
                color = COLORS[int(category) - 1]
            except IndexError:
                color = (255, 255, 255, 255)

            box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)
            box.setSize(l, w, h)
            box.translate(-l / 2, -w / 2, -h / 2)
            box.rotate(angle=rotation, x=0, y=0, z=1)
            box.translate(x, y, z)

            self.viewer.addItem(box)

            #################
            # heading lines #
            #################
            p1 = [-l / 2, -w / 2, -h / 2]
            p2 = [l / 2, -w / 2, h / 2]

            pts = np.array([p1, p2])
            l1 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
            l1.rotate(angle=rotation, x=0, y=0, z=1)
            l1.translate(x, y, z)

            self.viewer.addItem(l1)

            p3 = [-l / 2, -w / 2, h / 2]
            p4 = [l / 2, -w / 2, -h / 2]

            pts = np.array([p3, p4])

            l2 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
            l2.rotate(angle=rotation, x=0, y=0, z=1)
            l2.translate(x, y, z)

            self.viewer.addItem(l2)

            distance = np.linalg.norm([x, y, z], axis=0)
            self.boxes[distance] = (box, l1, l2)


    def show_pointcloud(self, filename: str) -> None:
        self.reset_viewer()
        self.cb.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.toggle_extracted_fog_btn.setEnabled(False)
        self.toggle_simulated_fog_btn.setEnabled(False)
        self.toggle_simulated_fog_dense_btn.setEnabled(False)

        if self.file_name == filename and self.current_pc is not None:
            # reuse the current pointcloud if the filename stays the same
            pc = self.current_pc
        else:
            self.file_name = filename
            pc = self.load_pointcloud(filename)

        self.success = False
        min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > 1.75   # in m
        pc = pc[min_dist_mask, :]

        self.current_pc = copy.deepcopy(pc)
        self.fogless_pc = copy.deepcopy(pc)

        if 'extraction' in filename:
            intensity_mask = pc[:, 3] <= self.threshold
            pc = pc[intensity_mask, :]

        colors = self.get_colors(pc)

        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
        self.current_mesh = mesh

        if self.success:
            self.viewer.addItem(mesh)
            self.reset_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            if 'extraction' not in filename:
                self.toggle_extracted_fog_btn.setEnabled(True)
                self.toggle_simulated_fog_btn.setEnabled(True)
                self.toggle_simulated_fog_dense_btn.setEnabled(True)

        if self.extracted_fog and self.success and 'extraction' not in filename:
            self.toggle_simulated_fog_btn.setEnabled(False)
            self.toggle_simulated_fog_dense_btn.setEnabled(False)
            fog_points = self.load_fog_points()
            self.extracted_fog_pc = fog_points
            intensity_mask = fog_points[:, 3] <= self.threshold
            fog_points = fog_points[intensity_mask, :]
            fog_colors = self.get_colors(fog_points)
            fog = gl.GLScatterPlotItem(pos=np.asarray(fog_points[:, 0:3]), size=self.point_size, color=fog_colors)
            self.extracted_fog_mesh = fog
            self.viewer.addItem(fog)

        if self.simulated_fog and self.success and 'extraction' not in filename:
            self.toggle_extracted_fog_btn.setEnabled(False)
            self.toggle_simulated_fog_dense_btn.setEnabled(False)
            self.reset_viewer()
            pc, simulated_fog_pc, info_dict = simulate_fog(self.p, self.current_pc, self.noise, self.gain,
                                                           self.noise_variant)
            self.simulated_fog_pc = simulated_fog_pc
            self.min_fog_response = info_dict['min_fog_response']
            self.max_fog_response = info_dict['max_fog_response']
            self.num_fog_responses = info_dict['num_fog_responses']
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
            self.viewer.addItem(mesh)

        if self.simulated_fog_dense and self.success and 'extraction' not in filename:
            self.toggle_simulated_fog_btn.setEnabled(False)
            self.toggle_extracted_fog_btn.setEnabled(False)
            self.reset_viewer()

            B = BetaRadomization(beta=float(self.p.alpha), seed=0)
            B.propagate_in_time(10)

            arguments = Namespace(sensor_type='Velodyne HDL-64E S3D', fraction_random=0.05)
            n_features = pc.shape[1]

            pc = haze_point_cloud(pc, B, arguments)
            pc = pc[:, :n_features]

            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
            self.viewer.addItem(mesh)

        distance = np.linalg.norm(pc[:, 0:3], axis=1)

        try:
            self.p.r_range = max(distance)
            self.log_string(pc=pc)
        except ValueError:
            self.p.r_range = 0

        if self.dataset == 'DENSE':
            self.cb.setEnabled(True)
            self.populate_dense_boxes(filename)
        else:
            self.cb.setCurrentText('all')
            self.cb.setEnabled(False)

        if self.boxes:

            try: # if heading lines are available
                for box_distance, (box, l1, l2) in self.boxes.items():
                    if box_distance < self.p.r_range:
                        self.viewer.addItem(box)
                        self.viewer.addItem(l1)
                        self.viewer.addItem(l2)
            except TypeError:
                for box_distance, box in self.boxes.items():
                    if box_distance < self.p.r_range:
                        self.viewer.addItem(box)

        if self.result_dict and self.show_predictions:

            self.visualize_predictions()


    def populate_dense_boxes(self, filename):

        root = str(Path.home()) + '/repositories/PCDet/lib/SeeingThroughFog/tools/DatasetViewer/calibs'
        tf_tree = 'calib_tf_tree_full.json'

        name_camera_calib = 'calib_cam_stereo_left.json'

        rgb_calib = load_calib_data(root, name_camera_calib, tf_tree)

        camera_to_velodyne_rgb = rgb_calib[1]

        label_path = Path(filename).parent.parent / 'gt_labels' / 'cam_left_labels_TMP'

        recording = Path(filename).stem                 # here without '.txt' as it will be added in read_label function
        label_file = os.path.join(label_path, recording)
        label = read_label(label_file, label_path, camera_to_velodyne=camera_to_velodyne_rgb)

        size = QtGui.QVector3D(1, 1, 1)

        self.boxes = {}

        # create annotation boxes
        for annotation in label:

            if annotation['identity'] in ['PassengerCar', 'Pedestrian', 'RidableVehicle']:

                x = annotation['posx_lidar']
                y = annotation['posy_lidar']
                z = annotation['posz_lidar']

                if annotation['identity'] == 'PassengerCar':
                    color = COLORS[0]
                elif annotation['identity'] == 'Pedestrian':
                    color = COLORS[1]
                else:
                    color = COLORS[2]

                distance = np.sqrt(x**2 + y**2 + z**2)

                box = gl.GLBoxItem(size, color=color)
                box.setSize(annotation['length'], annotation['width'], annotation['height'])
                box.translate(-annotation['length'] / 2, -annotation['width'] / 2, -annotation['height'] / 2)
                box.rotate(angle=-annotation['rotz'] * 180 / 3.14159265359, x=0, y=0, z=1)
                box.rotate(angle=-annotation['roty'] * 180 / 3.14159265359, x=0, y=1, z=0)
                box.rotate(angle=-annotation['rotx'] * 180 / 3.14159265359, x=1, y=0, z=0)
                box.translate(0, 0, annotation['height'] / 2)
                box.translate(x, y, z)

                self.boxes[distance] = box



    def log_string(self, pc: np.ndarray) -> None:

        log_string = f'max_dist ' + f'{int(self.p.r_range)}'.rjust(3, ' ') + ' m | ' + \
                     f'intensity [ ' + f'{int(min(pc[:, 3]))}'.rjust(3, ' ') + \
                     f', ' + f'{int(max(pc[:, 3]))}'.rjust(3, ' ') + ']' + ' ' + \
                     f'median ' + f'{int(np.round(np.median(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                     f'mean ' + f'{int(np.round(np.mean(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                     f'std ' + f'{int(np.round(np.std(pc[:, 3])))}'.rjust(3, ' ')

        if self.num_fog_responses > 0:
            range_fog_response_string = f'fog [ ' + f'{int(self.min_fog_response)}'.rjust(3, ' ') + \
                                        f', ' + f'{int(self.max_fog_response)}'.rjust(3, ' ') + ']'
            num_fog_responses_string = f'num_soft ' + f'{int(self.num_fog_responses)}'.rjust(6, ' ')
            num_remaining_string = f'num_hard ' + f'{int(len(self.current_pc) - self.num_fog_responses)}'.rjust(6, ' ')

            log_string = log_string + ' | ' + \
                         range_fog_response_string + ' ' + num_fog_responses_string + ' ' + num_remaining_string

        self.log_info.setText(log_string)


    def get_colors(self, pc: np.ndarray) -> np.ndarray:

        # create colormap
        if self.color_feature == 0:

            self.success = True
            feature = pc[:, 0]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 1:

            self.success = True
            feature = pc[:, 1]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 2:

            self.success = True
            feature = pc[:, 2]
            min_value = -1.5
            max_value = 0.5

        elif self.color_feature == 3:

            self.success = True
            feature = pc[:, 3]
            min_value = 0
            max_value = 255

        elif self.color_feature == 4:

            self.success = True
            feature = np.linalg.norm(pc[:, 0:3], axis=1)

            try:
                min_value = np.min(feature)
                max_value = np.max(feature)
            except ValueError:
                min_value = 0
                max_value = np.inf

        elif self.color_feature == 5:

            self.success = True
            feature = np.arctan2(pc[:, 1], pc[:, 0]) + np.pi
            min_value = 0
            max_value = 2 * np.pi

        else:  # self.color_feature == 6:

            try:
                feature = pc[:, 4]
                self.success = True

            except IndexError:
                feature = pc[:, 3]

            min_value = self.min_value
            max_value = self.max_value

        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

        if self.color_feature == 5:
            cmap = cm.hsv  # cyclic
        else:
            cmap = cm.jet  # sequential

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

        return colors


    def load_fog_points(self, index: int = None) -> np.ndarray:

        if index is None and self.extracted_fog_index == -1:
            index = RNG.integers(low=0, high=len(self.extracted_fog_file_list), size=1)[0]
            self.extracted_fog_index = index

        filename = self.extracted_fog_file_list[self.extracted_fog_index]
        fog_points = np.fromfile(filename, dtype=self.d_type)

        return fog_points.reshape((-1, 5))


    def load_pointcloud(self, filename: str) -> np.ndarray:

        self.reset_custom_values()

        if 'KITTI' in filename:
            self.set_kitti()

        if 'DENSE' in filename:
            self.set_dense()

        if 'nuScenes' in filename:
            self.set_nuscenes()

        if 'Lyft' in filename:
            self.set_lyft()

        if 'Waymo' in filename:
            self.set_waymo()

        if 'Honda' in filename:
            self.set_honda()

        if 'A2D2' in filename:
            self.set_audi()

        if 'PandaSet' in filename:
            self.set_panda()

        if 'Apollo' in filename:
            self.set_apollo()

        if 'Argoverse' in filename:
            self.set_argo()

        self.color_name = self.color_dict[self.color_feature]
        self.color_label.setText(self.color_name)

        # /srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/nuScenes/sweeps/LIDAR_TOP
        # /srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/nuScenes/lidarseg/v1.0-trainval

        if self.extension == 'ply':

            pc = self.load_from_ply(filename)

        elif self.extension == 'npz':

            pc = self.load_from_npz(filename)

        elif 'pkl' in self.extension:

            pc = self.load_from_pkl(filename)

        else:   # assume bin file

            pc = np.fromfile(filename, dtype=self.d_type)
            pc = pc.reshape((-1, self.num_features))

        pc[:,3] = np.round(pc[:,3] * self.intensity_multiplier)

        if self.dataset == 'Honda3D':

            self.file_name_label.setText(f'{Path(filename).parent.name}/'
                                         f'{Path(filename).name}')

        elif self.dataset == 'PandaSet' or self.dataset == 'Apollo':

            self.file_name_label.setText(f'{Path(filename).parent.parent.name}/'
                                         f'{Path(filename).parent.name}/'
                                         f'{Path(filename).name}')

        else:

            self.file_name_label.setText(str(Path(filename).name))

        return pc


    def load_from_pkl(self, filename: str) -> np.ndarray:
        if filename.endswith('gz'):
            with gzip.open(filename, 'rb') as f:
                data = pkl.load(f)
        else:
            with open(filename, 'rb') as f:
                data = pkl.load(f)
        if self.dataset == 'PandaSet':
            pc = data.drop(columns=['t']).values
        else:
            pc = data.values
        return pc


    def load_from_ply(self, filename: str) -> np.ndarray:
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)

        pc = np.array(plydata.elements[0].data.tolist())[:]

        if self.dataset == 'Honda3D':
            pc = np.delete(pc, [3, 4, 5, 6, 7, 8, 9, 12], 1)
        elif self.dataset == 'Argoverse':
            pc = pc
        else:
            pc = np.delete(pc, [4, 5, 6], 1)

        return pc


    def load_from_npz(self, filename: str) -> np.ndarray:
        npz = np.load(filename)
        pc_dict = {}
        for key in npz.keys():
            pc_dict[key] = npz[key]
        pc = None
        if self.dataset == 'A2D2':
            pc = np.column_stack((pc_dict['points'],
                                  pc_dict['reflectance'],
                                  pc_dict['lidar_id']))
        return pc


    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        logging.debug("enter")
        mimeData = e.mimeData()
        mimeList = mimeData.formats()
        filename = None

        if "text/uri-list" in mimeList:
            filename = mimeData.data("text/uri-list")
            filename = str(filename, encoding="utf-8")
            filename = filename.replace("file://", "").replace("\r\n", "").replace("%20", " ")
            filename = Path(filename)

        if filename.exists() and (filename.suffix == ".bin" or
                                  filename.suffix == ".ply" or
                                  filename.suffix == ".pickle"):
            e.accept()
            self.droppedFilename = filename
            self.extension = filename.suffix.replace('.', '')
        else:
            e.ignore()
            self.droppedFilename = None

    def dropEvent(self, e: QDropEvent) -> None:
        if self.droppedFilename:
            self.create_file_list(Path(self.droppedFilename).parent, self.droppedFilename)


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    app = QtGui.QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()

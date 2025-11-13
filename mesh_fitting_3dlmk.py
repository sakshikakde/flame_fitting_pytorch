import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import os
from argparse import Namespace
from model.flame import FLAME
from utils.image_utils import extract_mediapipe_lmk3d, plot_landmarks, align_with_flame_axes
from utils.mesh_utils import save_landmarks_as_ply, write_simple_obj
import cv2

class MeshFitting:
    def __init__(self, config, device='cpu', save_dir='./output'):

        self.config = config
        self.device = device
        self.flame_model = FLAME(self.config).to(device)


        # weights
        weights = {}
        # landmark term
        weights['lmk']   = 1.0   
        # shape regularizer (weight higher to regularize face shape more towards the mean)
        weights['shape'] = 1e-3
        # expression regularizer (weight higher to regularize facial expression more towards the mean)
        weights['expr']  = 1e-3
        # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
        weights['pose']  = 1e-2
        # number of shape and expression parameters (we do not recommend using too many parameters for fitting to sparse keypoints)
        self.weights = weights

        bz = 1
        self.shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(device))
        self.exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(device))
        self.pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(device))
        self.transl = nn.Parameter(torch.zeros(bz, 3).float().to(device))   
        self.scale = nn.Parameter(torch.tensor([1.0]).float().to(device))

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.optim_scale = torch.optim.Adam(
            [self.scale],
            lr=1e-1,
        )

        self.optim_rigid = torch.optim.Adam(
            [self.transl, self.scale],
            lr=1e-1,
        )

    def landmark_loss(self, pred_lmk, target_lmk):
        return torch.mean(torch.sum((pred_lmk - target_lmk)**2, dim=2))

    def optimize_rigid(self, target_lmk, num_iters=1000):
        for iter in range(num_iters):
            self.optim_rigid.zero_grad()

            vertices, pred_lmk = self.flame_model(
                shape_params=self.shape,
                expression_params=self.exp,
                pose_params=self.pose,
                transl=self.transl
            )
            pred_lmk = pred_lmk * self.scale

            lmk_loss = self.landmark_loss(pred_lmk, target_lmk) * self.weights['lmk']
            total_loss = lmk_loss

            total_loss.backward()

            self.optim_rigid.step()
            if iter % 10 == 0:
                print(f"Iter {iter}/{num_iters} - Total Loss: {total_loss.item():.4f} - Lmk Loss: {lmk_loss.item():.4f}")



    
    def save_final_mesh(self, mesh_save_path="fitted_mesh.ply", lmk_save_path="fitted_landmarks.ply"):
        vertices, landmarks = self.flame_model(
            shape_params=self.shape,
            expression_params=self.exp,
            pose_params=self.pose,
            transl=self.transl
        )
        vertices = vertices * self.scale
        landmarks = landmarks * self.scale
        vertices = vertices.squeeze().cpu().detach().numpy()
        landmarks = landmarks.squeeze().cpu().detach().numpy()
        faces = self.flame_model.faces

        write_simple_obj(mesh_v=vertices, mesh_f=faces, filepath=mesh_save_path, verbose=False)
        save_landmarks_as_ply(landmarks, lmk_save_path)

    


    def fit(self, image):
        # get landmarks 3D
        lmk3d = extract_mediapipe_lmk3d(image)
        plot_landmarks(image, lmk3d, save_file=os.path.join(self.save_dir, "mediapipe_landmarks.png"))
        lmk3d = align_with_flame_axes(lmk3d)
        

        static_lmk3d_idx = self.flame_model.lmk_idx.detach().numpy()
        static_lmk3d = lmk3d[static_lmk3d_idx, :]

        save_landmarks_as_ply(static_lmk3d, os.path.join(self.save_dir, "mediapipe_landmarks.ply"))

        self.optimize_rigid(torch.tensor(static_lmk3d).unsqueeze(0).float().to(self.device), num_iters=500)
        self.save_final_mesh(
            mesh_save_path=os.path.join(self.save_dir, "fitted_mesh.obj"),
            lmk_save_path=os.path.join(self.save_dir, "fitted_landmarks.ply"),
        )



config = {
    # FLAME model paths
    "flame_model_path": "/home/sakshi/projects/mesh_fitting/FLAME_PyTorch/model/flame2023.pkl",
    "use_mediapipe_embedding":True,
    "mediapipe_embedding_path": "/home/sakshi/projects/mesh_fitting/data/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz",
    "static_landmark_embedding_path": "/home/sakshi/projects/mesh_fitting/data/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz",
    "dynamic_landmark_embedding_path": "/home/sakshi/projects/mesh_fitting/FLAME_PyTorch/model/flame_dynamic_embedding.npy",

    # FLAME hyperparameters
    "shape_params": 100,
    "expression_params": 50,
    "pose_params": 6,

    # Training hyperparameters
    "use_face_contour": True,
    "use_3D_translation": True,   # False for RingNet project
    "optimize_eyeballpose": True, # False for RingNet project
    "optimize_neckpose": True,    # False for RingNet project

    "num_worker": 4,
    "batch_size": 1,
    "ring_margin": 0.5,
    "ring_loss_weight": 1.0,
}
args = Namespace(**config)
MeshFitting_obj = MeshFitting(args, device='cpu', save_dir='./output')
image = cv2.imread('/home/sakshi/projects/mesh_fitting/data/sakshi.png')
MeshFitting_obj.fit(image)  
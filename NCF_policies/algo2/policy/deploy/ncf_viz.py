import numpy as np
import pyvista as pv
from matplotlib import cm
from pyvistaqt import BackgroundPlotter
import trimesh
import tkinter as tk
import queue
import copy
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from vedo import trimesh2vedo
from vedo import Plotter, Mesh, Points, Text2D


class NCFViz:
    def __init__(
        self, off_screen: bool = False, zoom: float = 1.0, window_size: int = 0.5
    ):
        pv.global_theme.multi_rendering_splitting_position = 0.7
        """
            subplot(0, 0) main viz
            subplot(0, 1): tactile image viz
            subplot(1, 1): tactile codebook viz 
        """
        shape, row_weights, col_weights = ((2, 3), [0.8, 0.2], [0.3, 0.3, 0.4])
        groups = [(0, 0), (0, 1), (0, 2), (1, 0), (1, np.s_[1:2])]

        w, h = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()

        self.plotter = BackgroundPlotter(
            title="NCF",
            lighting="three lights",
            window_size=(int(w * window_size), int(h * window_size)),
            off_screen=off_screen,
            shape=shape,
            row_weights=row_weights,
            col_weights=col_weights,
            groups=groups,
            border_color="white",
            toolbar=False,
            menu_bar=False,
            auto_update=True,
        )
        self.zoom = zoom

        self.viz_queue = queue.Queue(1)
        self.plotter.add_callback(self.update_viz, interval=100)  # 50
        self.pause = False
        self.font_size = int(20 * window_size)

        self.cam_obj_1 = dict(
            position=(0.192824, 4.66691e-3, -0.249805),
            focal_point=(-4.82266e-3, -2.87693e-3, 5.10574e-7),
            viewup=(0.655673, -0.564232, 0.501733),
            distance=0.318628,
            clipping_range=(0.143946, 0.500430),
        )
        self.cam_obj_2 = dict(
            position=(-0.100776, -0.139449, -0.271581),
            focal_point=(3.92765e-3, -2.10310e-3, -3.81815e-3),
            viewup=(-0.904989, 0.398268, 0.149592),
            distance=0.318628,
            clipping_range=(0.165546, 0.532542),
        )

    def init_variables(self, object_mesh_path, object_pointcloud):
        self.object_pointcloud = np.load(object_pointcloud)
        self.n_points = self.object_pointcloud.shape[0]

        object_trimesh = trimesh.load_mesh(object_mesh_path)
        T = np.eye(4)
        T[0:3, 0:3] = Rot.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
        object_trimesh = object_trimesh.apply_transform(T)

        # self.object_mesh = trimesh2vedo(object_trimesh).clone()
        # self.object_mesh.subdivide(n=5, method=2)
        self.object_mesh = pv.wrap(object_trimesh)
        self.object_cloud = pv.PolyData(self.object_pointcloud)

        self.plotter.subplot(0, 0)
        self.plotter.add_text(
            "Realsense camera",
            position="upper_left",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="rs_title",
        )
        self.plotter.camera_position = "xy"
        self.plotter.camera.zoom(2)
        # self.plotter.camera_set = True

        self.plotter.subplot(0, 1)
        self.plotter.camera.Zoom(1)
        self.plotter.add_text(
            "DIGIT and VAE images",
            position="upper_left",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="digit_title",
        )

        self.plotter.subplot(0, 2)
        self.plotter.camera.zoom(1.0)
        self.plotter.add_text(
            "NCF contact patch estimation",
            position="upper_left",
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="ncf_title",
        )
        # self.plotter.camera_position = "yx"
        # self.plotter.camera.azimuth = 30
        # self.plotter.camera.Zoom(3.0)
        self.plotter.camera.position = self.cam_obj_1["position"]
        self.plotter.camera.focal_point = self.cam_obj_1["focal_point"]
        self.plotter.camera.up = self.cam_obj_1["viewup"]
        self.plotter.camera.distance = self.cam_obj_1["distance"]
        self.plotter.camera.clipping_range = self.cam_obj_1["clipping_range"]
        self.plotter.camera.Zoom(1.0)
        self.plotter.camera_set = True

        self.images = {"im": [], "path": []}
        self.image_plane = None

    def update_viz(self):
        if self.viz_queue.qsize():
            (
                info,
                rs_img,
                digit_left,
                digit_right,
                vae_left,
                vae_right,
                contact_vector,
                idx_query,
                image_savepath,
            ) = self.viz_queue.get()

            self.viz_rs_image(rs_img)
            self.viz_digit_images(digit_left, digit_right, vae_left, vae_right)
            self.viz_ncf_contact(contact_vector, idx_query)
            # self.viz_info(info)

            if image_savepath:
                self.images["im"].append(self.plotter.screenshot())
                self.images["path"].append(image_savepath)

            self.viz_queue.task_done()

    def update(
        self,
        info,
        rs_img,
        digit_left,
        digit_right,
        vae_left,
        vae_right,
        contact_vector,
        idx_query,
        image_savepath,
    ):
        if self.viz_queue.full():
            self.viz_queue.get()
        self.viz_queue.put(
            (
                info,
                rs_img,
                digit_left,
                digit_right,
                vae_left,
                vae_right,
                contact_vector,
                idx_query,
                image_savepath,
            ),
            block=False,
        )

    def viz_rs_image(self, rs_imgs):
        front = rs_imgs[0]
        back = rs_imgs[1]

        white = [255, 255, 255]
        front = cv2.copyMakeBorder(
            front, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        back = cv2.copyMakeBorder(
            back, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        rs_img = np.concatenate((front, back), axis=0)

        s = 10.0
        # if self.image_plane is None:
        image_plane = pv.Plane(
            direction=(0.0, 0.0, 1.0),
            i_size=rs_img.shape[1] * s,
            j_size=rs_img.shape[0] * s,
            i_resolution=rs_img.shape[1] - 1,
            j_resolution=rs_img.shape[0] - 1,
        )
        # self.image_plane
        image_plane.points[:, -1] = 0.25

        self.plotter.subplot(0, 0)
        # self.plotter.view_xy()
        # self.plotter.camera.Zoom(2.0)

        # image_tex = pv.numpy_to_texture(cv2.flip(rs_img, 1))
        image_tex = pv.numpy_to_texture(rs_img)
        self.plotter.add_mesh(
            image_plane,
            texture=image_tex,
            smooth_shading=False,
            show_scalar_bar=False,
            name="rs_img",
            render=False,
        )
        self.plotter.view_xy()
        self.plotter.camera.Zoom(1.5)

    def viz_digit_images(self, digit_left, digit_right, vae_left, vae_right):
        img_left = (cv2.resize(digit_left, (240, 320)) * 255).astype(np.uint8)
        img_right = (cv2.resize(digit_right, (240, 320)) * 255).astype(np.uint8)
        img_left_vae = (cv2.resize(vae_left, (240, 320)) * 255).astype(np.uint8)
        img_right_vae = (cv2.resize(vae_right, (240, 320)) * 255).astype(np.uint8)

        white = [255, 255, 255]
        img_left = cv2.copyMakeBorder(
            img_left, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        img_right = cv2.copyMakeBorder(
            img_right, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        img_left_vae = cv2.copyMakeBorder(
            img_left_vae, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        img_right_vae = cv2.copyMakeBorder(
            img_right_vae, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white
        )

        img_left = cv2.putText(
            img_left,
            "Left",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        img_right = cv2.putText(
            img_right,
            "Right",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        img_left_vae = cv2.putText(
            img_left_vae,
            "Left VAE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        img_right_vae = cv2.putText(
            img_right_vae,
            "Right VAE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        img_org = np.concatenate((img_left, img_right), axis=1)
        img_vae = np.concatenate((img_left_vae, img_right_vae), axis=1)
        img_digits = np.concatenate((img_org, img_vae), axis=0)
        img_digits = cv2.flip(img_digits, 1)

        s = 1.0
        image_plane = pv.Plane(
            direction=(0.0, 0.0, 1.0),
            i_size=img_digits.shape[1] * s,
            j_size=img_digits.shape[0] * s,
            i_resolution=img_digits.shape[1] - 1,
            j_resolution=img_digits.shape[0] - 1,
        )
        # self.image_plane
        image_plane.points[:, -1] = 0.25

        self.plotter.subplot(0, 1)
        image_tex = pv.numpy_to_texture(img_digits)
        self.plotter.add_mesh(
            image_plane,
            texture=image_tex,
            smooth_shading=False,
            show_scalar_bar=False,
            name="digit_img",
            render=False,
        )
        self.plotter.view_xy()
        self.plotter.camera.zoom(1.5)

    # def viz_ncf_contact(self, contact_vector, idx_query):
    #     pred_full = np.zeros((self.n_points))
    #     pred_full[idx_query] = contact_vector.cpu().numpy()

    #     pc_pred = Points(self.object_pointcloud, r=5)
    #     pc_pred = pc_pred.cmap(
    #         "plasma",
    #         pred_full,
    #         vmin=0.0,
    #         vmax=1.0,
    #     )

    #     mesh_pred = self.object_mesh.clone()
    #     mesh_pred = mesh_pred.interpolate_data_from(
    #         pc_pred, n=3, on="points", kernel="shepard"
    #     ).cmap("plasma", vmin=0.0, vmax=1.0)

    #     plt = Plotter(
    #         shape=[1, 2],
    #         axes=0,
    #         sharecam=False,
    #         title="NCF mugs",
    #         offscreen=True,
    #     )

    #     pred_name = Text2D("NCF - view 1", s=0.9)
    #     plt.show([mesh_pred, pc_pred, pred_name], at=0, camera=self.cam_obj_1)
    #     pred_name = Text2D("NCF - view 2", s=0.9)
    #     plt.show([mesh_pred, pc_pred, pred_name], at=1, camera=self.cam_obj_2)
    #     img_ncf = plt.screenshot(asarray=True)
    #     plt.close()
    #     img_ncf = cv2.flip(img_ncf, 1)
    #     # increase resolution

    #     scale = 1.3
    #     new_shape = (int(img_ncf.shape[1] * scale), int(img_ncf.shape[0] * scale))
    #     img_ncf = cv2.resize(img_ncf, new_shape, interpolation=cv2.INTER_AREA)

    #     s = 1.0
    #     image_plane = pv.Plane(
    #         direction=(0.0, 0.0, 1.0),
    #         i_size=img_ncf.shape[1] * s,
    #         j_size=img_ncf.shape[0] * s,
    #         i_resolution=img_ncf.shape[1] - 1,
    #         j_resolution=img_ncf.shape[0] - 1,
    #     )
    #     # self.image_plane
    #     image_plane.points[:, -1] = 0.25

    #     self.plotter.subplot(0, 2)
    #     image_tex = pv.numpy_to_texture(img_ncf)
    #     self.plotter.add_mesh(
    #         image_plane,
    #         texture=image_tex,
    #         smooth_shading=False,
    #         show_scalar_bar=False,
    #         name="ncf_img",
    #         render=False,
    #     )
    #     self.plotter.view_xy()
    #     # self.plotter.camera.zoom(1.6)

    def viz_ncf_contact(self, contact_vector, idx_query):
        pred_full = np.zeros((self.n_points))
        pred_full[idx_query] = contact_vector.cpu().numpy()

        contact_cloud = copy.deepcopy(self.object_cloud)
        contact_cloud["val"] = pred_full

        ncf_contact_mesh = self.object_mesh.interpolate(
            contact_cloud,
            # n_points=10,
            # sharpness=1.0,
            sharpness=2.0,
            strategy="null_value",
            radius=self.object_mesh.length / 15,
        )
        dargs = dict(
            cmap=cm.get_cmap("plasma"),
            scalars="val",
            interpolate_before_map=True,
            ambient=0.3,
            opacity=1.0,
            clim=[0.0, 1.0],
            show_scalar_bar=False,
            silhouette=False,
            reset_camera=False,
        )
        self.plotter.subplot(0, 2)
        self.plotter.add_mesh(ncf_contact_mesh, **dargs)
        self.plotter.add_mesh(contact_cloud, **dargs)

    def viz_info(self, info):
        self.plotter.subplot(1, 0)
        self.plotter.add_text(
            "Test {0}".format(info["episode"]),
            position=(50, 90),
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="episode text",
            render=True,
        )
        self.plotter.add_text(
            "Step {0}".format(info["step"]),
            position=(50, 40),
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="step text",
            render=True,
        )
        self.plotter.subplot(1, 1)
        self.plotter.add_text(
            "Policy: \n {0}".format(info["policy"]),
            position=(50, 70),
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="policy text",
            render=True,
        )

        self.plotter.subplot(1, 2)
        self.plotter.add_text(
            "Mug close to cupholder: ",
            position=(50, 90),
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="mug_close text",
            render=True,
        )
        self.plotter.add_text(
            "Yes" if info["is_mug_close"] else "No",
            # "{0}".format(info["is_mug_close"]),
            position=(370, 90),
            color="green" if info["is_mug_close"] else "red",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="mug_close result",
            render=True,
        )

        self.plotter.subplot(1, 2)
        self.plotter.add_text(
            "Mug correctly oriented: ",
            position=(50, 40),
            color="black",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="mug_orn text",
            render=True,
        )
        self.plotter.add_text(
            "Yes" if info["is_mug_oriented"] else "No",
            # "{0}".format(info["is_mug_oriented"]),
            position=(370, 40),
            color="green" if info["is_mug_oriented"] else "red",
            shadow=True,
            font="times",
            font_size=self.font_size,
            name="mug_orn result",
            render=True,
        )

    def save_imgs(self):
        if len(self.images):
            print("Saving images...")
            for im, path in zip(self.images["im"], self.images["path"]):
                im = Image.fromarray(im.astype("uint8"), "RGB")
                im.save(path)
            self.images = {"im": [], "path": []}

    def close(self):
        self.plotter.close()

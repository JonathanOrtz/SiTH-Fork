import os 
import math

import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
# import kaolin as kal
# import nvdiffrast
# from load_obj import load_obj
from torch.utils.data import DataLoader, Dataset, IterableDataset
# import multiprocessing as mp
import cv2
# Set the multiprocessing start method to 'spawn'
# mp.set_start_method('spawn')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import numpy as  np


class TestDiffDataset(Dataset):
    def __init__(self, args,root, device, size=512):
        self.root = root
        self.img_size = size

        # self.glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        self.device=device
        self.args=args
        # _, _, texv, texf, _ = load_obj(uv_template, load_materials=True)

        # self.texv = texv.to(self.device)
        # self.texf = texf.to(self.device)

        # self.front_folder = os.path.join(self.root, 'images')
        # self.smplx_foldser = os.path.join(self.root, 'smplx')
        
        # RGBA
        self.subject_list = self.get_subjects(self.root)
        # smplx obj
        # self.smplx_list = [os.path.join(self.smplx_folder, x) for x in sorted(os.listdir(self.smplx_folder)) if x.endswith('.obj')]


        # assert len(self.subject_list) == len(self.smplx_list)

        self.num_subjects = len(self.subject_list)
        if self.args.create_normals:
            from render import Render

            self.render = Render(size=512, device=device)
            if '/RENDER' in self.root:
                smpl_path=self.root.replace('/RENDER','/SMPL')
            elif '/RENDER_NORMAL' in self.root:
                smpl_path=self.root.replace('/RENDER_NORMAL','/SMPL')
            self.generate_smpl_normal(smpl_path, self.args.init, self.args.end)


        #  set camera
        look_at = torch.zeros( (1, 3), dtype=torch.float32, device=device)
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device)
        camera_position = torch.tensor( [[0, 0, -3]], dtype=torch.float32, device=device)


        # self.camera = kal.render.camera.Camera.from_args(eye=camera_position,
        #                                  at=look_at,
        #                                  up=camera_up_direction,
        #                                  width=self.img_size, height=self.img_size,
        #                                  near=-512, far=512,
        #                                 fov_distance=1.0, device=device)


        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba_mask= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  ])


        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.transform_clip_mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
    def augment_random_background(self, image, mask):
    # Random background
        bg_color = torch.rand(3) * 2 - 1
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        return image * mask + bg * (1 - mask)

    def  get_subjects(self,training_images_list_file):
        all_folder=os.listdir(training_images_list_file)
        paths=[]
        for fold in all_folder:
            sub_folder=os.listdir(os.path.join(training_images_list_file,fold))
            for sub_fold in sub_folder:
                if '/RENDER_NORMAL/' in training_images_list_file:
                    training_images_list_file_2=training_images_list_file
                    training_images_list_file_mod=training_images_list_file_2.replace('/RENDER_NORMAL/','/RENDER/')
                    if os.path.exists(os.path.join(training_images_list_file_mod,fold,sub_fold)):
                        # for i in range(0,36):

                            # if os.path.exists(os.path.join(training_images_list_file,fold,sub_fold,'k{}.person_rend_full.jpg'.format(i))):
                        paths.append(os.path.join(training_images_list_file,fold,sub_fold,os.listdir(os.path.join(training_images_list_file,fold,sub_fold))[0]))
                            # else:
                            #     paths.append(os.path.join(training_images_list_file,fold,sub_fold,'k{}.person_rend_full.png'.format(i)))
                else:
                    for el in os.listdir(os.path.join(training_images_list_file,fold,sub_fold)):
                        paths.append(os.path.join(training_images_list_file,fold,sub_fold,el))
        # paths[-3:]
        return paths

    def render_uv_map(self, camera, V, F, texv, texf, size):

        vertices_camera = camera.extrinsics.transform(V)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                            vertices_camera, F)
        face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera,unit=True)[..., -1:].contiguous()
        proj = camera.projection_matrix()[0:1]
        homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )[..., None]

        vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
        rast = nvdiffrast.torch.rasterize(
            self.glctx, vertices_clip, F.int(),
            (size, size), grad_db=False
        )
        rast0 = torch.flip(rast[0], dims=(1,))
        hard_mask = rast0[:, :, :, -1:] != 0

        uv_map = nvdiffrast.torch.interpolate(
            texv.cuda(), rast0, texf[...,:3].int().cuda()
        )[0] % 1.

        return uv_map
    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_rgb_image()
    def generate_smpl_normal(self,smpl_path_orig, init, end):
        smpl_list=os.listdir(smpl_path_orig)
        for i in range(init, end):
            smpl_path=os.path.join(smpl_path_orig,smpl_list[i])
            if  os.path.isdir(smpl_path):
                smpl_path_fin=os.path.join(smpl_path,'mesh_smplx.obj')
                smpl_verts, smpl_faces = load_obj(smpl_path_fin)
                # folder=os.path.split(smpl_path)[0]
                vmin = smpl_verts.min(0)
                vmax = smpl_verts.max(0)
                vmin=vmin.values.numpy()
                vmax=vmax.values.numpy()
                up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
        
            # vmed = np.median(smpl_verts, 0)
            # vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
            # y_scale = 180/(vmax[up_axis] - vmin[up_axis])up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
            
            
                y_scale = 180/(vmax[up_axis] - vmin[up_axis])
                vmed = np.median(smpl_verts.numpy(), 0)

                vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
                vmed = vmed[ np.newaxis,:]
                smpl_verts = smpl_verts - torch.Tensor(vmed)
            # print('important', vmed, y_scale)
            # scale=y_scale* torch.ones(3)
            # print(scale)
                smpl_verts=smpl_verts*y_scale
            
            
    # Add the translation vector to the matrix
            # print(smpl_verts.shape,vmed.shape)
            
            
            # 525
                condition=int(os.path.split(smpl_path)[1])
            # print(vmin,vmax,subject_name)
            

                theta_init=math.radians(270)
                cos_theta = math.cos(theta_init)
                sin_theta = math.sin(theta_init)
            
                rotation_matrix_x = torch.tensor([
                    [1, 0, 0],
                    [0, cos_theta, -sin_theta],
                    [0, sin_theta, cos_theta]
                ]).to(self.device)
                if condition>525:
                    smpl_verts_rot=torch.matmul(smpl_verts.to(self.device),rotation_matrix_x.T)
                else:
                    smpl_verts_rot=smpl_verts.to(self.device)
                theta_init=math.radians(180)
                cos_theta = math.cos(theta_init)
                sin_theta = math.sin(theta_init)

                rotation_matrix_y = torch.tensor([
                    [cos_theta, 0, sin_theta],
                    [0, 1, 0],
                    [-sin_theta, 0, cos_theta]
                ]).to(self.device)
                if condition>525:
                    smpl_verts_rot=torch.matmul(smpl_verts_rot,rotation_matrix_y.T)
                else:
                    smpl_verts_rot=smpl_verts.to(self.device)
            
            # if vmin.values[-1]<0 and vmax.values[-1]>0:
            #     condition=False
            # elif vmin.values[-1]>0 and vmax.values[-1]<0:
            #     condition=False
            # else:
            #     condition=True
                folder1=smpl_path.replace('/SMPL/','/SMPL_NORMAL/')
                folder2=smpl_path.replace('/SMPL/','/SMPL_MASK/')
                os.makedirs(folder1,exist_ok=True)
                os.makedirs(folder2,exist_ok=True)
            
                for vid in range(0,360,3):
                    save_F=os.path.join(folder1,'front_img_{}.jpg'.format(vid))
                    save_B=os.path.join(folder1,'back_img_{}.jpg'.format(vid))
                    save_R=os.path.join(folder1,'right_img_{}.jpg'.format(vid))
                    save_L=os.path.join(folder1,'left_img_{}.jpg'.format(vid))

                    save_mask_F=os.path.join(folder2,'front_mask_{}.jpg'.format(vid))
                    save_mask_B=os.path.join(folder2,'back_mask_{}.jpg'.format(vid))
                    save_mask_R=os.path.join(folder2,'right_mask_{}.jpg'.format(vid))
                    save_mask_L=os.path.join(folder2,'left_mask_{}.jpg'.format(vid))

                    if not  os.path.exists(save_R)  or not os.path.exists(save_L)or not os.path.exists(save_F)or not os.path.exists(save_B)or not os.path.exists(save_mask_F) or not os.path.exists(save_mask_R)or not os.path.exists(save_mask_L)or not os.path.exists(save_mask_B):
                        print(vid,save_F)
                        theta_init=math.radians(vid)
                        cos_theta = math.cos(theta_init)
                        sin_theta = math.sin(theta_init)
                        rotation_matrix_init = torch.tensor([
                            [cos_theta, 0, sin_theta],
                            [0, 1, 0],
                            [-sin_theta, 0, cos_theta]
                        ]).to(self.device)
                        smpl_verts_init = torch.matmul(smpl_verts_rot.to(self.device), rotation_matrix_init.T)

                        normal_front, normal_back = self.render_normal(
                                    # smpl_verts_init * torch.tensor([1.0, -1.0, -1.0]).to(self.device),
                                    smpl_verts_init.to(self.device),
                                    smpl_faces)
                        normal_front_mask, normal_back_mask = self.render.get_silhouette_image()

                        theta = math.radians(270)  # 旋转90度
                        cos_theta = math.cos(theta)
                        sin_theta = math.sin(theta)

                        rotation_matrix = torch.tensor([
                            [cos_theta, 0, sin_theta],
                            [0, 1, 0],
                            [-sin_theta, 0, cos_theta]
                        ]).to(self.device)
                        rotated_verts = torch.matmul(smpl_verts_init, rotation_matrix.T)

                        normal_right, normal_left=self.render_normal(rotated_verts *
                            torch.tensor([1.0, 1.0, 1.0]).to(self.device),smpl_faces)

                        normal_right_mask, normal_left_mask = self.render.get_silhouette_image()
                # print(normal_front_mask.shape)
                        image_F = (0.5 * (1.0 + normal_front[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
                        image_B = (0.5 * (1.0 + normal_back[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
                        image_R = (0.5 * (1.0 + normal_right[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
                        image_L = (0.5 * (1.0 + normal_left[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
                        #MEGLIO SE LE SALVO E LE RILEGGO CON IMAGE PER LE CONVENTION!!!
                        mask_F = normal_front_mask[0].detach().cpu().numpy() * 255.0
                        mask_B = normal_back_mask[0].detach().cpu().numpy() * 255.0
                        mask_R = normal_right_mask[0].detach().cpu().numpy() * 255.0
                        mask_L = normal_left_mask[0].detach().cpu().numpy() * 255.0
                
                # image_F=Image.fromarray(image_F)
                # image_B=Image.fromarray(image_B)
                # image_R=Image.fromarray(image_R)
                # image_L=Image.fromarray(image_L)
                        
                        

                        cv2.imwrite(save_F,image_F)
                        cv2.imwrite(save_B,image_B)
                        cv2.imwrite(save_R,image_R)
                        cv2.imwrite(save_L,image_L)
                        # tgt_uv = torch.zeros((self.img_size, self.img_size, 3))
                        cv2.imwrite(save_mask_F,mask_F)
                        cv2.imwrite(save_mask_B,mask_B)
                        cv2.imwrite(save_mask_R,mask_R)
                        cv2.imwrite(save_mask_L,mask_L)

    def load_smpl(self,smpl_path,smpl_param,vid):

        folder=os.path.split(smpl_path)[0]
        folder1=folder.replace('/SMPL/','/SMPL_NORMAL/')
        folder2=folder.replace('/SMPL/','/SMPL_MASK/')

        save_F=os.path.join(folder1,'front_img_{}.jpg'.format(vid))
        save_B=os.path.join(folder1,'back_img_{}.jpg'.format(vid))
        save_R=os.path.join(folder1,'right_img_{}.jpg'.format(vid))
        save_L=os.path.join(folder1,'left_img_{}.jpg'.format(vid))

        save_mask_F=os.path.join(folder2,'front_mask_{}.jpg'.format(vid))
        save_mask_B=os.path.join(folder2,'back_mask_{}.jpg'.format(vid))
        save_mask_R=os.path.join(folder2,'right_mask_{}.jpg'.format(vid))
        save_mask_L=os.path.join(folder2,'left_mask_{}.jpg'.format(vid))
        
        
        normal_front =Image.open(save_F).convert('RGB')
        normal_front_mask = Image.open(save_mask_F).convert('L')
        normal_back = Image.open(save_B).convert('RGB')
        normal_back_mask = Image.open(save_mask_B).convert('L')
        normal_right =Image.open(save_R).convert('RGB')
        normal_right_mask= Image.open(save_mask_R).convert('L')
        normal_left =Image.open(save_L).convert('RGB')
        normal_left_mask =Image.open(save_mask_L).convert('L')
        
        return normal_front, normal_front_mask, normal_back, normal_back_mask, normal_right, normal_right_mask, normal_left, normal_left_mask
    
    def __getitem__(self, index):


        # Load images, missing ground truth one and eventually normals
        image_path=self.subject_list[index]
        extract_vid=os.path.basename(image_path)

        # vid_orig=int(extract_vid.split('_')[0])
        # view_ids=[]
        # path_orig=os.path.dirname(image_path)
        # if vid_orig<90:
        #     view_ids.append(vid_orig+180)
            
        #     view_ids.append(vid_orig+90)
        #     view_ids.append(360+(vid_orig-90))
        # elif vid_orig>=270:
        #     view_ids.append(vid_orig-180) 
        #     view_ids.append((vid_orig+90)-360)
            
        #     view_ids.append(vid_orig-90) 
        # elif vid_orig>=180 and vid_orig<270:
        #     view_ids.append(vid_orig-180)

        #     view_ids.append(vid_orig+90)
        #     view_ids.append(vid_orig-90)

        # elif vid_orig<180 and vid_orig>=90:
        #     view_ids.append(vid_orig+180)
        #     view_ids.append(vid_orig+90)
        #     view_ids.append(vid_orig-90)

        # final_list_clip=[]
        final_list_rgb=[]
        # for idx, ids in enumerate(view_ids):
        #     new_path=os.path.join(path_orig,'{}_0_00.jpg'.format(ids))

        #     if '/RENDER/' in new_path:
        #         mask_path=new_path.replace('/RENDER/','/MASK/')
        #         if not os.path.exists(mask_path):
        #             mask_path=mask_path[:-3]+'png'
        #         if not os.path.exists(mask_path):
        #             mask_path=mask_path[:-3]+'jpg'
        #     elif '/RENDER_NORMAL/' in new_path:
        #         mask_path=new_path.replace('/RENDER_NORMAL/','/MASK/')
        #         if not os.path.exists(mask_path):
        #             mask_path=mask_path[:-3]+'jpg'
        #         if not os.path.exists(mask_path):
        #             mask_path=mask_path[:-3]+'png'
        
        #     image = Image.open(new_path)
        #     mask = Image.open(mask_path).convert('L')
        #     if not image.mode == "RGB":
        #         image = image.convert("RGB")
        #     # mask=np.array(mask).astype
        #     # image.putalpha(mask)
        #     # image = np.array(image).astype(np.uint8)
        #     # image_clip = self.transform_clip(image)
        #     # image = (image/127.5 - 1.0).astype(nps.float32)
        #     # mask_array = np.array(mask)
        #     # mask_clip = self.transform_clip_mask(mask)
        #     # binary_mask = (mask_array > 0).astype(np.float32)  # Convert to binary mask (0 or 1)

        #     # Apply the mask to the normalized image
        #     # image = image * binary_mask[:, :, np.newaxis]src_clip_image = src_rgb * src_mask
        #     # src_clip_image = image_clip * mask_clip
        #     # final_list_clip.append(src_clip_image)

        #     image_rgb = self.transform_rgba(image)
        #     # image = (image/127.5 - 1.0).astype(nps.float32)
        #     # mask_array = np.array(mask)
        #     mask_rgb = self.transform_rgba_mask(mask)
        #     # binary_mask = (mask_array > 0).astype(np.float32)  # Convert to binary mask (0 or 1)

        #     # Apply the mask to the normalized image
        #     # image = image * binary_mask[:, :, np.newaxis]src_clip_image = src_rgb * src_mask
        #     src_rgb_image = image_rgb * mask_rgb
        #     final_list_rgb.append(src_rgb_image)
        #     if idx==0:
        #         back_mask=mask_rgb.clone()
        # stacked_images_gt = torch.cat(final_list_rgb, dim=0)
        # stacked_images_clip = torch.cat(final_list_clip, dim=2)
        # final_array = np.reshape(stacked_images, (512, 512, -1))
        # print('important', final_array.shape)



        new_path=self.subject_list[index]
        
        if '/RENDER/' in new_path:
            mask_path=new_path.replace('/RENDER/','/MASK_PARTIAL/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            mask_path=new_path.replace('/RENDER_NORMAL/','/MASK_PARTIAL/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
        image = Image.open(new_path)
        mask = Image.open(mask_path).convert('L')
        if not image.mode == "RGB":
            image = image.convert("RGB")



            
        src_rgb = self.transform_clip(Image.open(new_path))
        mask_clip = self.transform_clip_mask(mask)
        # src_mask = src_rgba[-1:,...]
        # src_rgb = src_rgba[:-1,...]
        src_clip_image = src_rgb * mask_clip



        # src_rgba = self.transform_rgba(Image.open(new_path))
        # if '/RENDER/' in new_path:
        #     mask_path=new_path.replace('/RENDER/','/MASK/')
        #     if not os.path.exists(mask_path):
        #         mask_path=mask_path[:-3]+'png'
        #     if not os.path.exists(mask_path):
        #         mask_path=mask_path[:-3]+'jpg'
        # elif '/RENDER_NORMAL/' in new_path:
        #     mask_path=new_path.replace('/RENDER_NORMAL/','/MASK/')
        #     if not os.path.exists(mask_path):
        #         mask_path=mask_path[:-3]+'jpg'
        #     if not os.path.exists(mask_path):
        #         mask_path=mask_path[:-3]+'png'

        image = Image.open(new_path)
        mask = Image.open(mask_path).convert('L')
        if not image.mode == "RGB":
            image = image.convert("RGB")
                
        image_rgb = self.transform_rgba(image)
            # image = (image/127.5 - 1.0).astype(nps.float32)
            # mask_array = np.array(masask)
        mask_rgb = self.transform_rgba_mask(mask)
        src_image = image_rgb * mask_rgb

        print('pd',src_image.shape)


        image_new = Image.open(new_path)

        if '/RENDER/' in new_path:
            mask_path=new_path.replace('/RENDER/','/MASK_COMBINED/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            mask_path=new_path.replace('/RENDER_NORMAL/','/MASK_COMBINED/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
        mask_new = Image.open(mask_path).convert('L')
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        image_rgb_new = self.transform_rgba(image_new)
            # image = (image/127.5 - 1.0).astype(nps.float32)
            # mask_array = np.array(masask)
        mask_rgb_new = self.transform_rgba_mask(mask_new)
        target = image_rgb_new * mask_rgb_new


        # # load smpl verts
        # if '/RENDER/' in path_orig:
        #     smpl_path=path_orig.replace('/RENDER/','/SMPL/') + '/mesh_smplx.obj'
        #     smpl_param=path_orig.replace('/RENDER/','/SMPL/') + '/smplx_param.pkl'
        # elif '/RENDER_NORMAL/' in path_orig:
        #     smpl_path=path_orig.replace('/RENDER_NORMAL/','/SMPL/') + '/mesh_smplx.obj'
        #     smpl_param=path_orig.replace('/RENDER_NORMAL/','/SMPL/') + '/smplx_param.pkl'


        if '/RENDER/' in new_path:
            mask_smpl_path=new_path.replace('/RENDER/','/SMPL_MASK/')
            if not os.path.exists(mask_smpl_path):
                mask_smpl_path=mask_smpl_path[:-3]+'png'
            if not os.path.exists(mask_path):
                mask_smpl_path=mask_smpl_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            mask_smpl_path=new_path.replace('/RENDER_NORMAL/','/SMPL_MASK/')
            if not os.path.exists(mask_smpl_path):
                mask_smpl_path=mask_smpl_path[:-3]+'jpg'
            if not os.path.exists(mask_smpl_path):
                mask_smpl_path=mask_smpl_path[:-3]+'png'

        if '/RENDER/' in new_path:
            mask_normal_path=new_path.replace('/RENDER/','/SMPL_NORMAL/')
            if not os.path.exists(mask_normal_path):
                mask_normal_path=mask_normal_path[:-3]+'png'
            if not os.path.exists(mask_normal_path):
                mask_normal_path=mask_normal_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            mask_normal_path=new_path.replace('/RENDER_NORMAL/','/SMPL_NORMAL/')
            if not os.path.exists(mask_normal_path):
                mask_normal_path=mask_normal_path[:-3]+'jpg'
            if not os.path.exists(mask_smpl_path):
                mask_normal_path=mask_normal_path[:-3]+'png'

        if '/RENDER/' in new_path:
            mask_path=new_path.replace('/RENDER/','/MASK_DIFFERENCE/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            mask_path=new_path.replace('/RENDER_NORMAL/','/MASK_DIFFERENCE/')
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'jpg'
            if not os.path.exists(mask_path):
                mask_path=mask_path[:-3]+'png'
        mask_diff = Image.open(mask_path).convert('L')
        # mask_diff[mask_diff<127]=0
        # mask_diff[mask_diff>=127]=255
        mask_rgb_diff = self.transform_rgba_mask(mask_diff)
        # mask_rgb_diff[mask_rgb_diff<127]=0
        # mask_rgb_diff[mask_rgb_diff>=127]=255
        all_zeros = (mask_rgb_diff == 0).all()
        # if all_zeros.item():
        if torch.count_nonzero(mask_rgb_diff)<50:
            print('no diffusion needed:', torch.count_nonzero(mask_rgb_diff),mask_rgb_diff.shape)
            yes_diff=False
        else:
            print('diffusion needed:', torch.count_nonzero(mask_rgb_diff),mask_rgb_diff.shape)
            yes_diff=True
        
        normal_left = Image.open(mask_normal_path)
        normal_left_mask = Image.open(mask_smpl_path).convert('L')

        if '/RENDER/' in new_path:
            math_path=new_path.replace('/RENDER/','/MATRICES/')
            if not os.path.exists(math_path):
                math_path=math_path[:-3]+'npy'
            # if not os.path.exists(mask_path):
            #     mask_path=mask_path[:-3]+'jpg'
        elif '/RENDER_NORMAL/' in new_path:
            math_path=math_path.replace('/RENDER_NORMAL/','/MATRICES/')
            if not os.path.exists(mask_path):
                math_path=math_path[:-3]+'npy'
            # if not os.path.exists(mask_path):
            #     mask_path=mask_path[:-3]+'png'

        matrices=np.load(math_path)
        # normal_front, normal_front_mask, normal_back, normal_back_mask, normal_right, normal_right_mask, normal_left, normal_left_mask = self.load_smpl(smpl_path,smpl_param,vid_orig)
        # uv = self.render_uv_map(self.camera, V.to(self.device), F.to(self.device), self.texv, self.texf, self.img_size)
        
        # normal_front_mask = self.transform_rgba_mask(normal_front_mask)
        # normal_front = normal_front * normal_front_mask
        normal_left_mask = self.transform_rgba_mask(normal_left_mask)
        normal_left = self.transform_rgba(normal_left)
        normal_left = normal_left * normal_left_mask
        # normal_back_mask = self.transform_rgba_mask(normal_back_mask)
        # normal_back = self.transform_rgba(normal_back)
        # normal_back = normal_back * normal_back_mask
        # normal_right_mask = self.transform_rgba_mask(normal_right_mask)
        # normal_right = self.transform_rgba(normal_right)
        # normal_right = normal_right  * normal_right_mask

        #back_mask=torch.flip(mask_rgb, [2])

        masks_guide=normal_left_mask
        normals_guide=normal_left

        # tgt_image = self.augment_random_background(tgt_rgb, tgt_mask)

        # tgt_uv[..., :2] = uv[0].cpu()

        # view condition is always the same for back images
        view_cond = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)
        view_cond_90 = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)
        view_cond_270 = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)
        return {'src_ori_image': src_image,
                'src_image': src_clip_image,
                'target_img': target,
                # 'tgt_uv':  tgt_uv.permute(2,0,1) * 2. - 1,
                'M_crop': torch.from_numpy(matrices),
                'inpaint_mask': mask_rgb_diff,
                'tgt_normals':normals_guide,
                'tgt_mask': masks_guide,
                'yes_diff':yes_diff,
                'view_cond': view_cond,
                'filename':self.subject_list[index].split('/')[-3]+'/' +self.subject_list[index].split('/')[-2]+'/'+self.subject_list[index].split('/')[-1]
        }

    def __len__(self):
        return self.num_subjects
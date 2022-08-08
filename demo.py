import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import model as mdl
import json
from model.common import arange_pixels
from configloading import load_config


# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--gpu', type=int, default=-1, help='gpu')
parser.add_argument('--obj_name', type=str, default='bunny',)
parser.add_argument('--chunk', type=int, default=1024)


args = parser.parse_args()
cfg = load_config('data/config.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

for sdir in ['rgb', 'vis', 'specular', 'novel_view']:
    os.makedirs(os.path.join('out', sdir), exist_ok=True)

model = mdl.NeuralNetwork(cfg)
renderer = mdl.Renderer(model, cfg, device=device)
checkpoint_io = mdl.CheckpointIO('data', model=model)

try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
it = load_dict.get('it', 100000)

img_dir = os.path.join('dataset', args.obj_name)
para = json.load(open(os.path.join(img_dir, 'params.json')))
KK = np.array(para['K']).astype(np.float32)
poses = np.array(para['pose_c2w']).astype(np.float32)
h,w = para['imhw']

renderer.im_res = (h,w)
pose0 = poses.copy()
poses[:3,1:3]*=-1.
pose_ori = torch.tensor(poses.copy()).to(device)

light_pos = np.array(para['light_test']).astype(np.float32) 
poses_view = np.array(para['view_test']).astype(np.float32) 
poses_view[:,:3,1:3]*=-1.
light_view = pose_ori[:3,3].clone()

light_pos = torch.tensor(light_pos).to(device)
poses = torch.tensor(poses).to(device)
poses_view = torch.tensor(poses_view).to(device)
camera_mat = torch.tensor(KK).to(device)[None,]
scale_mat = torch.eye(4,dtype=torch.float32).to(device)

for di, data in enumerate(tqdm(light_pos, ncols=120)):
    p_loc, pixels = arange_pixels(resolution=(h, w))
    p_loc = p_loc.to(device)
    light_src = light_pos[di:di+1]
    world_mat = poses[None,]

    with torch.no_grad():
        norm_pred, mask_obj,vis_pred, depth_pred, rgb, albedo, spec = [],[],[],[],[],[],[]
        for ii, pixels_i in enumerate(torch.split(p_loc, args.chunk, dim=1)):
            out_dict = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                        add_noise=False, eval_=True, it=it, light_src=light_src, 
                        novel_view=False, view_ori=pose_ori[None,])
            norm_pred.append(out_dict.get('normal_pred',None))
            mask_obj.append(out_dict.get('mask_pred',None))
            depth_pred.append(out_dict['depth'])
            vis_pred.append(out_dict['vis'])
            rgb.append(out_dict.get('rgb_fine',None))
            albedo.append(out_dict.get('albedo_fine',None))
            spec.append(out_dict.get('specular_fine',None))
        norm_pred = torch.cat(norm_pred, dim=1).detach().cpu().numpy().reshape(w,h,3).transpose(1,0,2).astype(np.float32)
        mask_obj = torch.cat(mask_obj, dim=0).reshape(w,h).permute(1,0).detach().cpu().numpy()
        depth_pred = torch.cat(depth_pred, dim=1).reshape(w,h).permute(1,0).detach().cpu().numpy().astype(np.float32)
        vis_pred = torch.cat(vis_pred, dim=1).reshape(w,h).permute(1,0).detach().cpu().numpy().astype(np.float32)
        rgb = torch.cat(rgb, dim=1).detach().cpu().numpy().reshape(w,h,3).transpose(1,0,2).astype(np.float32)
        albedo = torch.cat(albedo, dim=1).reshape(w,h,3).permute(1,0,2).detach().cpu().numpy().astype(np.float32)
        spec = torch.cat(spec, dim=1).reshape(w,h,3).permute(1,0,2).detach().cpu().numpy().astype(np.float32)

        img = Image.fromarray((rgb.clip(0,1) * 255).round().astype(np.uint8))
        img.save(os.path.join('out/rgb/ld_{:03d}.png'.format(di+1)))
        img = Image.fromarray((vis_pred.clip(0,1) * 255).round().astype(np.uint8))
        img.save(os.path.join('out/vis/ld_{:03d}.png'.format(di+1)))
        img = Image.fromarray((spec.clip(0,1) * 255).round().astype(np.uint8))
        img.save(os.path.join('out/specular/ld_{:03d}.png'.format(di+1)))
        if di > 0: continue
        img = Image.fromarray((mask_obj * 255).astype(np.uint8))
        img.save(os.path.join('out/mask.png'))
        depth_pred = (depth_pred-depth_pred[mask_obj].min())/(depth_pred[mask_obj].max()-depth_pred[mask_obj].min())
        img = Image.fromarray((depth_pred.clip(0,1) * 255).astype(np.uint8))
        img.save(os.path.join('out/depth.png'))
        img = Image.fromarray((norm_pred.clip(-1,1) * 127.5+127.5).round().astype(np.uint8))
        img.save(os.path.join('out/normal.png'))
        img = Image.fromarray((albedo.clip(0,1) * 255).astype(np.uint8))
        img.save(os.path.join('out/albedo.png'))

for di, data in enumerate(tqdm(poses_view, ncols=120)):
    p_loc, pixels = arange_pixels(resolution=(h, w))
    p_loc = p_loc.to(device)
    light_src = light_view[None,]
    world_mat = poses_view[di:di+1]
    with torch.no_grad():
        surface = []
        for ii, pixels_i in enumerate(torch.split(p_loc, args.chunk, dim=1)):
            out_dict = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                        add_noise=False, eval_=True, it=it, light_src=light_src, 
                        novel_view=True, view_ori=pose_ori[None,])
            surface.append(out_dict.get('surface',None))
        surface = torch.cat(surface, dim=1).detach().cpu().numpy().reshape(w,h,3).transpose(1,0,2).astype(np.float32)
        img = Image.fromarray((surface.clip(0,1) * 255).round().astype(np.uint8))
        img.save(os.path.join(f'out/novel_view/view_{di+1:02d}.png'))

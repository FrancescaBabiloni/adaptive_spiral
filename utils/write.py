import trimesh
import pyvista as pv
import os
import glob
import shutil
import cv2
from trimesh.exchange.export import export_mesh
import trimesh
from tqdm import tqdm

def render_mesh(tri_mesh, op, ws = (2048,1152)):
    # Render the view from the given camera pose
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    #face
    pv_mesh = pv.wrap(tri_mesh)
    plotter.add_mesh(pv_mesh,lighting=True, smooth_shading=True)

    #set easy camera position
    plotter.camera_position= 'xy'
    plotter.reset_camera()

    #save  
    plotter.screenshot(op, transparent_background=False, window_size=ws)
    print(f'done, saved in: {op}')
    return
    
def export_trimesh_ply(mesh, op, name='m'):
    export_mesh(mesh, os.path.join(op, name + '.ply'))
    return

def export_databatch_ply(data, batch, frame, out_dir='.'):
    f = data.face[batch,frame].numpy().T
    v = data.x[batch,frame].numpy()
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    op = os.path.join(out_dir, f'exp_{data.e[batch]}_subject_{data.id[batch]}')
    if not os.path.exists(op): os.makedirs(op)
    export_trimesh_ply(mesh, op, name=f'f_{str(f).zfill(4)}')
    return op

def export_pred_ply(vertices, faces, frame, odir, name='tmp'):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if not os.path.exists(odir): os.makedirs(odir)
    export_trimesh_ply(mesh, odir, name=f'{name}_f_{str(frame).zfill(4)}')
    return 

def save_pv_mesh(ip, op, ws = (2048,1152) ):
    tri_mesh = trimesh.load(ip)
    render_mesh(tri_mesh, op, ws = ws)
    return  

def save_ply_as_images(file_list, out_dir, ws = (2048,1152)):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for _f in file_list:
        name = ''.join(_f.split('/')[-3:]).replace('ply', 'jpg')
        _of = os.path.join(out_dir, name)
        save_pv_mesh(_f, _of, ws = (2048,1152))

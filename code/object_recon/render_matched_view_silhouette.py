"""Render raw and train-mask-culled MILo meshes at matched held-out views.

The raw mesh is transformed into the same upright object frame as the delivered
candidate. Camera poses are transformed by the inverse object-frame transform.
Only held-out cameras/masks are used for display and metric reporting.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image


def load_mesh(path: Path, transform: np.ndarray | None = None):
    mesh=o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty(): raise ValueError(f"empty mesh: {path}")
    if transform is not None:
        v=np.asarray(mesh.vertices)
        mesh.vertices=o3d.utility.Vector3dVector((transform[:3,:3]@v.T).T+transform[:3,3])
    scene=o3d.t.geometry.RaycastingScene(nthreads=0,device=o3d.core.Device("CPU:0"))
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.triangles,dtype=np.uint32)
    scene.add_triangles(o3d.core.Tensor(vertices,dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces,dtype=o3d.core.Dtype.UInt32))
    colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    return scene,faces,colors


def render(scene, faces, colors, w2c, width,height,fx,fy,cx,cy):
    xs=np.arange(width,dtype=np.float32); ys=np.arange(height,dtype=np.float32)
    xx,yy=np.meshgrid(xs,ys)
    dirs=np.stack(((xx-cx)/fx,(yy-cy)/fy,np.ones_like(xx)),axis=-1).reshape(-1,3)
    R=w2c[:3,:3]; t=w2c[:3,3]
    origin=-R.T@t; world_dirs=dirs@R
    rays=np.concatenate((np.broadcast_to(origin,world_dirs.shape),world_dirs),axis=1)
    hit=scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32),dtype=o3d.core.Dtype.Float32))
    ids=hit['primitive_ids'].numpy(); valid=ids<len(faces)
    mask=valid.reshape(height,width)
    rgb=np.zeros((height*width,3),dtype=np.float32)
    if colors is not None and np.any(valid):
        tri=faces[ids[valid]]
        uv=hit['primitive_uvs'].numpy()[valid]
        bary=np.stack((1-uv[:,0]-uv[:,1],uv[:,0],uv[:,1]),axis=1)
        rgb[valid]=(colors[tri]*bary[...,None]).sum(axis=1)
        normals=hit['primitive_normals'].numpy()[valid]
        # Fixed camera-relative light, with two-sided Lambert shading for an open surface.
        light=-np.array([0.25,-0.25,1.0]); light/=np.linalg.norm(light)
        diffuse=np.abs(normals@light)
        rgb[valid]*=(0.38+0.62*diffuse[:,None])
    else:
        rgb[valid]=0.78
    return np.clip(rgb.reshape(height,width,3),0,1),mask


def boundary(mask):
    m=mask.astype(bool)
    eroded=m.copy()
    eroded[1:] &= m[:-1]; eroded[:-1] &= m[1:]
    eroded[:,1:] &= m[:,:-1]; eroded[:,:-1] &= m[:,1:]
    return m & ~eroded


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-mesh',type=Path,required=True)
    ap.add_argument('--candidate-mesh',type=Path,required=True)
    ap.add_argument('--object-frame',type=Path,required=True)
    ap.add_argument('--cameras-json',type=Path,required=True)
    ap.add_argument('--validation-dir',type=Path,required=True)
    ap.add_argument('--views',nargs=2,default=['1979_000048.jpg','1979_000070.jpg'])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    meta=json.loads(args.cameras_json.read_text())
    frame=json.loads(args.object_frame.read_text())
    T_obj_col=np.asarray(frame['T_object_from_colmap'],dtype=np.float64)
    T_col_obj=np.asarray(frame['T_colmap_from_object'],dtype=np.float64)
    raw_scene,raw_faces,raw_colors=load_mesh(args.raw_mesh,T_obj_col)
    cand_scene,cand_faces,cand_colors=load_mesh(args.candidate_mesh)
    indexed={v['name']:v for v in meta['views']}
    fig,axes=plt.subplots(2,2,figsize=(10,7.4),dpi=220)
    rows=[('Raw learned-SDF',raw_scene,raw_faces,raw_colors),
          ('Training-mask culled',cand_scene,cand_faces,cand_colors)]
    records=[]
    rendered={}
    for col,name in enumerate(args.views):
        key=name if name.endswith('.jpg') else name+'.jpg'
        view=indexed[key]
        w2c_col=np.asarray(view['world_to_camera'],dtype=np.float64)
        w2c_obj=w2c_col@T_col_obj
        gt_img=Image.open(args.validation_dir/(Path(key).stem+'.png')).convert('RGBA')
        gt=np.asarray(gt_img)[...,3]>=128
        outs=[]; ious=[]
        for row,(label,scene,faces,colors) in enumerate(rows):
            rgb,pred=render(scene,faces,colors,w2c_obj,meta['width'],meta['height'],
                            meta['fx'],meta['fy'],meta['cx'],meta['cy'])
            inter=np.count_nonzero(pred[::2,::2]&gt[::2,::2]); union=np.count_nonzero(pred[::2,::2]|gt[::2,::2])
            iou=inter/union if union else 1.0
            records.append({'view':key,'model':label,'silhouette_iou_stride2':iou})
            ious.append(iou)
            rendered[(row,col)]=(rgb,pred)
            outs.append(pred)
        union=np.logical_or.reduce([gt,*outs])
        yy,xx=np.where(union)
        pad=max(12,int(max(yy.max()-yy.min()+1,xx.max()-xx.min()+1)*0.10))
        y0=max(0,int(yy.min())-pad); y1=min(meta['height'],int(yy.max())+pad+1)
        x0=max(0,int(xx.min())-pad); x1=min(meta['width'],int(xx.max())+pad+1)
        for row,(label,_,_,_) in enumerate(rows):
            rgb,pred=rendered[(row,col)]
            panel=rgb[y0:y1,x0:x1].copy()
            # The same yellow contour is the held-out foreground reference in both rows.
            gt_edge=boundary(gt)[y0:y1,x0:x1]
            panel[gt_edge]=[1.0,0.80,0.10]
            axes[row,col].imshow(panel)
            title=f'{Path(key).stem}\n{label} · IoU {ious[row]:.3f}' if row==0 else f'{label} · IoU {ious[row]:.3f}'
            axes[row,col].set_title(title)
            axes[row,col].axis('off')
        axes[0,col].set_xlabel(key.replace('.jpg',''))
        axes[0,col].xaxis.set_label_position('top'); axes[0,col].xaxis.label.set_size(10)
    for row,(label,_,_,_) in enumerate(rows): axes[row,0].set_ylabel(label,fontsize=10)
    fig.suptitle('Same held-out cameras; yellow contour = reference silhouette',fontsize=12,y=.99)
    fig.tight_layout(rect=[0,0,1,.96])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(json.dumps({'figure':str(args.output),'records':records},indent=2))

if __name__=='__main__': main()

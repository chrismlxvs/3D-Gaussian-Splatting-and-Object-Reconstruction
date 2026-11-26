"""Build a one-way camera rotation about the fixed plush target."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def unit(v):
    n=np.linalg.norm(v)
    if n < 1e-10: raise ValueError("degenerate direction")
    return v/n


def rotate(v, axis, angle):
    axis=unit(axis)
    return v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+axis*np.dot(axis,v)*(1-np.cos(angle))


def pose_at(c2w0, pivot, up, angle):
    A=np.column_stack([rotate(np.eye(3)[:,i],up,angle) for i in range(3)])
    c2w=np.eye(4)
    c2w[:3,:3]=A@c2w0[:3,:3]
    c2w[:3,3]=pivot+A@(c2w0[:3,3]-pivot)
    return np.linalg.inv(c2w)


def project(world, w2c, intr):
    cam=w2c[:3,:3]@world+w2c[:3,3]
    return [float(intr['fx']*cam[0]/cam[2]+intr['cx']),
            float(intr['fy']*cam[1]/cam[2]+intr['cy'])]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--garden-cameras',type=Path,required=True)
    p.add_argument('--placement',type=Path,required=True)
    p.add_argument('--view',default='DSC07956.JPG')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--render-keys-output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=240)
    p.add_argument('--render-keys',type=int,default=120)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--sweep-degrees',type=float,default=15.0)
    p.add_argument('--rotation-sign',choices=('auto','negative','positive'),default='auto',
                   help='auto selects the wider vase separation; negative sweeps toward the vase from the current view')
    a=p.parse_args()
    cams=json.loads(a.garden_cameras.read_text()); placement=json.loads(a.placement.read_text())
    source=next(v for v in cams['views'] if v['name']==a.view)
    w0=np.asarray(source['world_to_camera'],dtype=np.float64); c2w0=np.linalg.inv(w0)
    C0=c2w0[:3,3]; pivot=np.asarray(placement['mesh_center_scene'],dtype=np.float64)
    up=unit(np.asarray(placement['garden_axes']['up'],dtype=np.float64))
    target_px=np.asarray(placement['target_pixel']['xy'],dtype=float)
    normal=np.asarray(placement['contact_plane']['normal'],dtype=float); off=float(placement['contact_plane']['offset'])
    # A stable table reference near the vase chooses the orbit direction that increases separation from the plush.
    px,py=650.0,380.0
    ray_cam=np.array([(px-cams['cx'])/cams['fx'],(py-cams['cy'])/cams['fy'],1.0])
    ray_world=c2w0[:3,:3]@ray_cam
    denom=float(normal@ray_world)
    if abs(denom)<1e-8: raise ValueError('vase reference ray parallel to table')
    t=-(float(normal@C0)+off)/denom
    vase=C0+t*ray_world
    sweep=np.deg2rad(a.sweep_degrees)
    sign_scores={}
    for sign in (-1.0,1.0):
        pxv=project(vase,pose_at(c2w0,pivot,up,sign*sweep),cams)
        sign_scores[str(int(sign))]=float(np.linalg.norm(np.asarray(pxv)-target_px))
    if a.rotation_sign=='negative':
        direction=-1.0
    elif a.rotation_sign=='positive':
        direction=1.0
    else:
        direction=max((-1.0,1.0),key=lambda s:sign_scores[str(int(s))])
    def build(count, fps, key=False):
        views=[]; errors=[]
        for i in range(count):
            t=i/(count-1) if count>1 else 0.0
            eased=t*t*(3-2*t)
            w=pose_at(c2w0,pivot,up,direction*sweep*eased)
            name=f'frame_{i:03d}.png'
            uv=project(pivot,w,cams); errors.append(float(np.linalg.norm(np.asarray(uv)-target_px)))
            views.append({'name':name,'world_to_camera':w.tolist()})
        return views, errors
    full,errs=build(a.frames,a.fps)
    out={**{k:cams[k] for k in ('width','height','fx','fy','cx','cy')},'views':full,
        'fps':a.fps,'duration_seconds':a.frames/a.fps,'motion':'one-way camera orbit/rotation around plush center; no return',
        'rotation_axis_scene':up.tolist(),'sweep_degrees_signed':float(direction*a.sweep_degrees),
        'object_transform_policy':'one static object_to_scene transform for all frames',
        'target_pixel':target_px.tolist(),'target_projection_error_max_px':max(errs),
        'reference_table_point_projection_first':project(vase,np.asarray(full[0]['world_to_camera']),cams),
        'reference_table_point_projection_last':project(vase,np.asarray(full[-1]['world_to_camera']),cams),
        'rotation_direction_selection':a.rotation_sign,
        'sign_candidate_separations_px':sign_scores,'source_view':a.view}
    keys, keyerrs=build(a.render_keys, a.fps*a.frames/a.render_keys, key=True)
    keydoc={**{k:cams[k] for k in ('width','height','fx','fy','cx','cy')},'views':keys,
        'fps':a.render_keys/a.duration_seconds if 'duration_seconds' in locals() else a.render_keys/(a.frames/a.fps),
        'duration_seconds':a.frames/a.fps,'source_full_path_frames':a.frames,
        'source_full_frame_indices':[int(round(i*(a.frames-1)/(a.render_keys-1))) for i in range(a.render_keys)],
        'note':'Rendered keyframes sampled uniformly from the full camera rotation path; final 24 fps clip interpolates between them.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.render_keys_output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    a.render_keys_output.write_text(json.dumps(keydoc,indent=2)+'\n')
    print(json.dumps({'full_frames':len(full),'render_keyframes':len(keys),'signed_sweep_degrees':direction*a.sweep_degrees,
        'object_center_projection_error_max_px':max(errs),'vase_pixel_first':out['reference_table_point_projection_first'],
        'vase_pixel_last':out['reference_table_point_projection_last'],'sign_candidate_separations_px':sign_scores},indent=2))

if __name__=='__main__': main()

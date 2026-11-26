"""Rasterize a colored plushie mesh and depth-composite it over Garden."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr


def project_clip(cam_vertices: torch.Tensor, intr, near: float):
    width,height,fx,fy,cx,cy=intr
    z=cam_vertices[:,2]
    safe=z.clamp(min=near)
    x=2*(fx*cam_vertices[:,0]/safe+cx)/width-1
    y=2*(fy*cam_vertices[:,1]/safe+cy)/height-1
    clip=torch.stack((x*safe,y*safe,safe-near,safe),dim=-1)
    invalid=torch.tensor([0.,0.,0.,-1.],device=clip.device)
    clip=torch.where((z>near)[:,None],clip,invalid)
    return clip,z


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--mesh",type=Path,required=True)
    p.add_argument("--placement",type=Path,required=True)
    p.add_argument("--camera-json",type=Path,required=True)
    p.add_argument("--garden-rgb",type=Path,required=True)
    p.add_argument("--garden-depth",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--physical-gpu",type=int,required=True)
    p.add_argument("--depth-tolerance",type=float,default=.02)
    p.add_argument("--table-plane-tolerance",type=float,default=.03,
                   help="scene-depth distance from the fitted tabletop considered a table hit")
    p.add_argument("--table-contact-band",type=float,default=.002,
                   help="keep this much mesh height at the support plane subject to normal depth occlusion")
    p.add_argument("--preserve-vertex-colors",action="store_true",
                   help="skip additional Lambert shading because captured mesh colors already include lighting")
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--debug-dir",type=Path)
    a=p.parse_args()
    if a.stride<1: raise ValueError("stride must be at least one")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("bind exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    device=torch.device("cuda:0"); torch.cuda.set_device(device)
    mesh=trimesh.load(a.mesh,force="mesh",process=False)
    if mesh.is_empty or not mesh.faces.size or mesh.visual.vertex_colors is None:
        raise ValueError("mesh must contain faces and vertex colors")
    placement=json.loads(a.placement.read_text(encoding="utf-8"))
    transform=np.asarray(placement["object_to_scene"],dtype=np.float32)
    plane=placement["contact_plane"]
    plane_normal=torch.as_tensor(plane["normal"],dtype=torch.float32,device=device)
    plane_offset=float(plane["offset"])
    cameras=json.loads(a.camera_json.read_text(encoding="utf-8"))
    intr=tuple(cameras[k] for k in ("width","height","fx","fy","cx","cy"))
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.faces,dtype=np.int32)
    world=vertices@transform[:3,:3].T+transform[:3,3]
    colors=np.asarray(mesh.visual.vertex_colors[:,:3],dtype=np.float32)/255.
    normals=np.asarray(mesh.vertex_normals,dtype=np.float32)@np.linalg.inv(transform[:3,:3])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True).clip(1e-8)
    if not a.preserve_vertex_colors:
        # Optional gentle outdoor fill. By default, keep the historical render;
        # use --preserve-vertex-colors when the captured RGB already contains
        # illumination and a second lighting pass makes the mesh look muddy.
        up=np.asarray(placement["garden_axes"]["up"],dtype=np.float32)
        front=np.asarray(placement["garden_axes"]["front"],dtype=np.float32)
        right=np.cross(up,front); right/=np.linalg.norm(right)
        light=up*.82+front*.32+right*.24; light/=np.linalg.norm(light)
        lambert=np.maximum(0,normals@light)
        colors=colors*(.78+.22*lambert[:,None])
    v_t=torch.from_numpy(world).to(device)
    c_t=torch.from_numpy(colors).to(device)
    z_t=torch.from_numpy(world).to(device)
    f_t=torch.from_numpy(faces).to(device=device,dtype=torch.int32)
    raster=dr.RasterizeCudaContext(device=device)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.debug_dir: a.debug_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for index in range(0,len(cameras["views"]),a.stride):
        view=cameras["views"][index]
        name=Path(view["name"]).stem
        garden_path=a.garden_rgb/f"{name}.png"
        depth_path=a.garden_depth/f"{name}.npz"
        if not garden_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"missing Garden RGB/depth for {name}")
        with Image.open(garden_path) as im: background=np.asarray(im.convert("RGB"),dtype=np.float32)/255.
        with np.load(depth_path) as z:
            garden_depth=z["depth"].astype(np.float32)
            garden_alpha=z["alpha"].astype(np.float32)
        w2c=np.asarray(view["world_to_camera"],dtype=np.float32)
        camera=world@w2c[:3,:3].T+w2c[:3,3]
        clip,cam_z=project_clip(torch.from_numpy(camera).to(device),intr,.01)
        rast,_=dr.rasterize(raster,clip[None],f_t,
                             resolution=(int(intr[1]),int(intr[0])),grad_db=False)
        mask=(rast[0,:,:,3]>0).float()[None,:,:,None]
        rgb,_=dr.interpolate(c_t[None],rast,f_t)
        world_pos,_=dr.interpolate(v_t[None],rast,f_t)
        mesh_z,_=dr.interpolate(torch.from_numpy(camera[:,2:3]).to(device)[None],rast,f_t)
        rgb=dr.antialias(rgb,rast,clip[None],f_t)[0].clamp(0,1)
        alpha=dr.antialias(mask,rast,clip[None],f_t)[0,:,:,0].clamp(0,1)
        depth=mesh_z[0,:,:,0]
        gd=torch.from_numpy(garden_depth).to(device)
        ga=torch.from_numpy(garden_alpha).to(device)
        depth_occluded=(ga>.10)&(gd>0)&(gd<depth-a.depth_tolerance)
        # Expected depth from splatted Gaussians can place a tabletop sample
        # slightly in front of an inserted object even when the mesh point is
        # above the fitted support plane. Back-project both depth maps and let
        # the mesh win only for those plane hits; real elevated occluders, such
        # as the vase, retain the regular per-pixel depth test.
        height,width=garden_depth.shape
        yy,xx=torch.meshgrid(torch.arange(height,device=device,dtype=torch.float32),
                             torch.arange(width,device=device,dtype=torch.float32),
                             indexing="ij")
        rays=torch.stack(((xx-intr[4])/intr[2],(yy-intr[5])/intr[3],
                          torch.ones_like(xx)),dim=-1)
        camera_center=np.linalg.inv(w2c.astype(np.float64))[:3,3]
        camera_center_t=torch.as_tensor(camera_center,dtype=torch.float32,device=device)
        rotation_t=torch.as_tensor(w2c[:3,:3],dtype=torch.float32,device=device)
        scene_points=(rays*gd[...,None])@rotation_t+camera_center_t
        scene_plane_distance=(scene_points*plane_normal).sum(dim=-1)+plane_offset
        # Interpolate the rasterized world position directly; reconstructing it
        # from pixel coordinates and camera depth can flip rows under raster
        # origin conventions and falsely classify the far side as below table.
        mesh_plane_distance=(world_pos[0]*plane_normal).sum(dim=-1)+plane_offset
        table_depth_hit=(ga>.10)&(gd>0)&(scene_plane_distance.abs()<=a.table_plane_tolerance)
        mesh_above_contact=mesh_plane_distance>a.table_contact_band
        table_depth_suppressed=depth_occluded&table_depth_hit&mesh_above_contact
        occluded=depth_occluded&~table_depth_suppressed
        visible=(alpha>.02)&(depth>.01)&(~occluded)
        weight=alpha*visible.float()
        bg=torch.from_numpy(background).to(device)
        comp=rgb*weight[...,None]+bg*(1-weight[...,None])
        result=(comp.clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)
        Image.fromarray(result,"RGB").save(a.output/f"{name}.png")
        visible_pixels=int(visible.sum().item())
        total_pixels=int((alpha>.02).sum().item())
        occluded_pixels=int(((alpha>.02)&occluded).sum().item())
        suppressed_pixels=int(((alpha>.02)&table_depth_suppressed).sum().item())
        mesh_pixel_mask=alpha>.02
        table_hit_pixels=int((mesh_pixel_mask&table_depth_hit).sum().item())
        above_contact_pixels=int((mesh_pixel_mask&table_depth_hit&mesh_above_contact).sum().item())
        mesh_plane_values=mesh_plane_distance[mesh_pixel_mask]
        mesh_plane_quantiles=torch.quantile(mesh_plane_values,
            torch.tensor([0.,.1,.5,.9,1.],device=device)) if mesh_plane_values.numel() else torch.zeros(5,device=device)
        summary.append({"index":index,"name":view["name"],"mesh_pixels":total_pixels,
                        "visible_mesh_pixels":visible_pixels,
                        "scene_occluded_mesh_pixels":occluded_pixels,
                        "table_depth_occlusion_suppressed_pixels":suppressed_pixels,
                        "table_plane_hit_mesh_pixels":table_hit_pixels,
                        "above_contact_table_hit_mesh_pixels":above_contact_pixels,
                        "mesh_plane_height_quantiles":mesh_plane_quantiles.cpu().tolist(),
                        "visible_fraction":visible_pixels/max(total_pixels,1)})
        if a.debug_dir:
            vis=(visible.cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(vis,"L").save(a.debug_dir/f"{name}_visible.png")
            occluded_img=(occluded.cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(occluded_img,"L").save(a.debug_dir/f"{name}_occluded.png")
            suppressed_img=(table_depth_suppressed.cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(suppressed_img,"L").save(a.debug_dir/f"{name}_table_suppressed.png")
        print(f"composited {index+1}/{len(cameras['views'])} {name}: "
              f"visible {visible_pixels:,}/{total_pixels:,}; "
              f"table-depth rejects {suppressed_pixels:,}",flush=True)
    report={"mesh":str(a.mesh),"placement":str(a.placement),
            "camera_json":str(a.camera_json),"garden_depth_render":str(a.garden_depth),
            "depth_tolerance":a.depth_tolerance,
            "table_plane_tolerance":a.table_plane_tolerance,
            "table_contact_band":a.table_contact_band,
            "physical_gpu":a.physical_gpu,
            "views":summary}
    (a.output/"composite_summary.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()

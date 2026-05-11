"""Retargeting visualization — two separate outputs per run:

  frame{N}_mano_2d.png / frame{N}_mano_3d.html
      MANO hand only: original vs contact-adjusted
      Shows: mesh, skeleton, contact vertices, centroids, DexPilot vectors

  frame{N}_xhand_2d.png / frame{N}_xhand_3d.html
      MANO + xhand robot: original vs contact-adjusted
      Shows: MANO mesh/skeleton, contact centroids, DexPilot vectors (both),
             xhand skeleton, xhand DexPilot vectors
      (contact point clouds omitted for clarity)

Usage:
    conda activate vjepa2-312
    python visualize_contact_retarget.py \
        --npz /path/to/rgb_hawor/retarget_input.npz \
        --frame 10
"""
import argparse, base64, os, sys, pickle as pkl

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as Rscipy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'third_party', 'dex-retargeting'))
from _paths import URDF_ROOT, CONFIG_DIR
from dex_retargeting.retargeting_config import RetargetingConfig

MANO_PKL = ('/virtual_lab/ljw_rvlab/youngbo/skill2policy/third_party/'
            'HACO_RELEASE/data/base_data/human_models/MANO_RIGHT.pkl')

R_MANO_XHAND = {
    "right": np.array([[0,  0, 1], [0, -1, 0], [1,  0, 0]], dtype=np.float32),
    "left":  np.array([[0, 0, -1], [0,  1, 0], [1,  0, 0]], dtype=np.float32),
}

_KPT_TO_FINGER = {0: 5}
for _f, _s in enumerate([1, 5, 9, 13, 17]):
    for _k in range(_s, _s + 4):
        _KPT_TO_FINGER[_k] = _f

_FINGERTIP_KPT = [4, 8, 12, 16, 20]
_FINGER_NAMES  = ['thumb', 'index', 'middle', 'ring', 'pinky']
_FINGER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

_MANO_BONES = [
    [0,1],[0,5],[0,9],[0,13],[0,17],
    [1,2],[2,3],[3,4], [5,6],[6,7],[7,8],
    [9,10],[10,11],[11,12], [13,14],[14,15],[15,16],
    [17,18],[18,19],[19,20],
]

_XHAND_CHAINS_RIGHT = [
    ['right_hand_link', 'right_hand_ee_link'],
    ['right_hand_ee_link', 'right_hand_index_bend_link',
     'right_hand_index_rota_link1', 'right_hand_index_rota_link2', 'right_hand_index_rota_tip'],
    ['right_hand_ee_link', 'right_hand_mid_link1',
     'right_hand_mid_link2', 'right_hand_mid_tip'],
    ['right_hand_ee_link', 'right_hand_ring_link1',
     'right_hand_ring_link2', 'right_hand_ring_tip'],
    ['right_hand_ee_link', 'right_hand_pinky_link1',
     'right_hand_pinky_link2', 'right_hand_pinky_tip'],
    ['right_hand_link', 'right_hand_thumb_bend_link',
     'right_hand_thumb_rota_link1', 'right_hand_thumb_rota_link2', 'right_hand_thumb_rota_tip'],
]


def xhand_chains(hand):
    return [[l.replace('right_hand', f'{hand}_hand') for l in c]
            for c in _XHAND_CHAINS_RIGHT]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def vertex_finger_labels(jw, vw):
    d = np.linalg.norm(vw[:, None] - jw[None, :], axis=2)
    return np.array([_KPT_TO_FINGER[k] for k in np.argmin(d, axis=1)], dtype=np.int32)


def to_canonical(jw, vw, root_orient, hand):
    Rr = Rscipy.from_rotvec(root_orient).as_matrix().astype(np.float32)
    R  = R_MANO_XHAND[hand]
    jc = (Rr.T @ (jw - jw[0:1]).T).T @ R
    vc = (Rr.T @ (vw - jw[0:1]).T).T @ R
    return jc, vc


def load_mano_faces():
    with open(MANO_PKL, 'rb') as f:
        return np.array(pkl.load(f, encoding='latin1')['f'], dtype=np.int64)


def compute_xhand_fk(robot, qpos):
    robot.compute_forward_kinematics(qpos)
    return {fr.name: robot.get_link_pose(i)[:3, 3]
            for i, fr in enumerate(robot.model.frames)
            if fr.type == pin.FrameType.BODY}


def load_qpos_from_pkl(pkl_path, frame_num, start_idx):
    """Load qpos for a single frame from a retargeting pkl file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f'pkl not found: {pkl_path}\n'
            f'  Run retarget_from_npz.py / retarget_from_npz_contact.py first.')
    with open(pkl_path, 'rb') as f:
        d = pkl.load(f)
    t = frame_num - start_idx
    return d['data'][t].astype(np.float32)


def load_scene(data, faces, hand, frame_num, contact_dir, pkl_dir):
    si   = int(data['start_idx'])
    t    = frame_num - si
    hidx = 0 if hand == 'left' else 1

    jw = data[f'joints_{hand}'][t].astype(np.float32)
    vw = data[f'verts_{hand}'][t].astype(np.float32)
    ro = data['mano_global_orient'][hidx][t].astype(np.float32)

    cf = np.load(os.path.join(contact_dir, f'rgb_frame{frame_num:05d}.npz'),
                 allow_pickle=False) if os.path.exists(
        os.path.join(contact_dir, f'rgb_frame{frame_num:05d}.npz')) else {}
    cmask = np.array(cf.get(f'{hand}_contact_mask', np.zeros(778, bool)))

    jmp, vmp = to_canonical(jw, vw, ro, hand)
    labels   = vertex_finger_labels(jw, vw)

    jc = jmp.copy()
    centroids = {}
    for fi, tk in enumerate(_FINGERTIP_KPT):
        ct = cmask & (labels == fi)
        if ct.any():
            c = vmp[ct].mean(0)
            jc[tk]        = c
            centroids[fi] = c

    # build robot for FK (no retargeting needed)
    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)
    cfg   = os.path.join(CONFIG_DIR, f'xhand_{hand}_dexpilot.yml')
    opt   = RetargetingConfig.load_from_file(cfg).build().optimizer
    robot = opt.robot
    idx   = opt.target_link_human_indices  # (2,15)

    # load qpos from pkl files
    qpos_orig    = load_qpos_from_pkl(
        os.path.join(pkl_dir, f'qpos_xhand_{hand}.pkl'), frame_num, si)
    qpos_contact = load_qpos_from_pkl(
        os.path.join(pkl_dir, f'qpos_xhand_contact_{hand}.pkl'), frame_num, si)

    return dict(
        jmp=jmp, jc=jc, vmp=vmp, faces=faces,
        cmask=cmask, labels=labels, centroids=centroids,
        idx=idx, opt=opt, hand=hand,
        xhand_orig=compute_xhand_fk(robot, qpos_orig),
        xhand_contact=compute_xhand_fk(robot, qpos_contact),
    )


def find_rgb(npz_path, frame_num):
    ep  = os.path.dirname(os.path.dirname(os.path.abspath(npz_path)))
    p   = os.path.join(ep, 'rgb', f'rgb_frame{frame_num:05d}.png')
    return p if os.path.exists(p) else None


def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# Plotly helpers
# ══════════════════════════════════════════════════════════════════════════════

def arrow(p0, p1, color, name=None, show=False, lg=None, tip=0.18, w=4):
    d = p1 - p0; t = p0 + (1 - tip) * d; g = lg or name
    return [
        go.Scatter3d(x=[p0[0],t[0]], y=[p0[1],t[1]], z=[p0[2],t[2]],
                     mode='lines', line=dict(color=color, width=w),
                     name=name, showlegend=show, legendgroup=g),
        go.Cone(x=[t[0]], y=[t[1]], z=[t[2]],
                u=[d[0]*tip], v=[d[1]*tip], w=[d[2]*tip],
                colorscale=[[0,color],[1,color]], showscale=False,
                sizemode='absolute', sizeref=0.004,
                name=name, showlegend=False, legendgroup=g),
    ]


SCENE_CFG = dict(
    bgcolor='#0d1117',
    xaxis=dict(backgroundcolor='#0d1117', gridcolor='#333', color='#aaa'),
    yaxis=dict(backgroundcolor='#0d1117', gridcolor='#333', color='#aaa'),
    zaxis=dict(backgroundcolor='#0d1117', gridcolor='#333', color='#aaa'),
    aspectmode='cube',
)


def html_wrap(plotly_div, title_str, legend_note, rgb_b64, rgb_name, frame_num):
    img_tag = ''
    if rgb_b64:
        img_tag = (f'<div style="text-align:center;margin:12px 0 6px">'
                   f'<p style="color:#aaa;font-size:12px;margin:0 0 4px">'
                   f'RGB frame {frame_num} — {rgb_name}</p>'
                   f'<img src="data:image/png;base64,{rgb_b64}" '
                   f'style="max-height:280px;border:1px solid #333;border-radius:4px"></div>')
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{title_str}</title>'
            f'<style>body{{background:#1a1a2e;color:white;font-family:sans-serif;'
            f'margin:0;padding:12px}}h2{{text-align:center;color:#ccc;'
            f'font-weight:300;margin:8px 0}}.note{{text-align:center;'
            f'font-size:12px;color:#aaa;margin-bottom:8px}}</style></head><body>'
            f'<h2>{title_str}</h2>'
            f'<div class="note">{legend_note}</div>'
            f'{img_tag}{plotly_div}</body></html>')


# ══════════════════════════════════════════════════════════════════════════════
# ① MANO-only visualization
# ══════════════════════════════════════════════════════════════════════════════

def _mano_traces(sc, j_use, show_contact, vcol, lg):
    vmp, faces = sc['vmp'], sc['faces']
    jo         = sc['jmp']
    cmask, lbl = sc['cmask'], sc['labels']
    centroids  = sc['centroids']
    idx        = sc['idx']
    traces     = []

    traces.append(go.Mesh3d(
        x=vmp[:,0], y=vmp[:,1], z=vmp[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='lightgray', opacity=0.18, name='MANO mesh',
        showlegend=True, legendgroup=f'{lg}_mesh',
        lighting=dict(ambient=0.6, diffuse=0.8),
    ))
    for a, b in _MANO_BONES:
        traces.append(go.Scatter3d(
            x=[jo[a,0],jo[b,0]], y=[jo[a,1],jo[b,1]], z=[jo[a,2],jo[b,2]],
            mode='lines', line=dict(color='#607d8b', width=2),
            showlegend=False, legendgroup=f'{lg}_skel'))
    traces.append(go.Scatter3d(
        x=jo[:,0], y=jo[:,1], z=jo[:,2], mode='markers',
        marker=dict(size=3.5, color='#78909c'),
        name='MANO joints', showlegend=True, legendgroup=f'{lg}_jts'))

    if show_contact and cmask.any():
        for fi in range(5):
            m = cmask & (lbl == fi)
            if m.any():
                cv = vmp[m]
                traces.append(go.Scatter3d(
                    x=cv[:,0], y=cv[:,1], z=cv[:,2], mode='markers',
                    marker=dict(size=3, color=_FINGER_COLORS[fi], opacity=0.8),
                    name=f'{_FINGER_NAMES[fi]} contact',
                    legendgroup=f'{lg}_ct{fi}'))

    tips = jo[_FINGERTIP_KPT]
    traces.append(go.Scatter3d(
        x=tips[:,0], y=tips[:,1], z=tips[:,2], mode='markers',
        marker=dict(size=8, color='#00bcd4', symbol='circle',
                    line=dict(color='white', width=1)),
        name='Original fingertip', showlegend=True, legendgroup=f'{lg}_tip'))

    if show_contact:
        for fi, c in centroids.items():
            orig = jo[_FINGERTIP_KPT[fi]]
            traces.append(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]], mode='markers',
                marker=dict(size=12, color=_FINGER_COLORS[fi], symbol='diamond',
                            line=dict(color='white', width=1)),
                name=f'{_FINGER_NAMES[fi]} centroid ★',
                showlegend=True, legendgroup=f'{lg}_cen{fi}'))
            traces += arrow(orig, c, 'white', name=f'{_FINGER_NAMES[fi]} shift',
                            show=True, lg=f'{lg}_sh{fi}')

    for i, (s, d) in enumerate(zip(idx[0], idx[1])):
        traces += arrow(j_use[s], j_use[d], vcol,
                        name='DexPilot vec', show=(i == 0), lg=f'{lg}_vec')
    return traces


def save_mano_3d(scenes, frame_num, rgb_path, out_path):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type':'scene'}]*2]*2,
        subplot_titles=[
            'Left — Original (HaWoR fingertips)',
            'Left — Contact-adjusted (HACO centroid)',
            'Right — Original (HaWoR fingertips)',
            'Right — Contact-adjusted (HACO centroid)',
        ],
        vertical_spacing=0.04, horizontal_spacing=0.02,
    )
    cfgs = [
        ('left',  False, '#4fc3f7', 'lo', 1, 1),
        ('left',  True,  '#ff7043', 'lc', 1, 2),
        ('right', False, '#4fc3f7', 'ro', 2, 1),
        ('right', True,  '#ff7043', 'rc', 2, 2),
    ]
    for hand, sc_flag, col, lg, r, c in cfgs:
        sc    = scenes[hand]
        j_use = sc['jc'] if sc_flag else sc['jmp']
        for tr in _mano_traces(sc, j_use, sc_flag, col, lg):
            fig.add_trace(tr, row=r, col=c)

    for k in ['scene','scene2','scene3','scene4']:
        fig.update_layout(**{k: SCENE_CFG})

    cr = [_FINGER_NAMES[i] for i in scenes['right']['centroids']]
    cl = [_FINGER_NAMES[i] for i in scenes['left']['centroids']]
    fig.update_layout(
        paper_bgcolor='#1a1a2e', font=dict(color='white'),
        title=dict(text=(f'MANO Retargeting | frame {frame_num}<br>'
                         f'<sup>Right: {", ".join(cr) or "none"}  |  '
                         f'Left: {", ".join(cl) or "none"}</sup>'),
                   font=dict(size=14, color='white'), x=0.5),
        legend=dict(bgcolor='rgba(20,20,40,0.7)', font=dict(size=9)),
        margin=dict(l=0,r=0,t=90,b=0), width=1600, height=900,
    )
    b64  = img_b64(rgb_path) if rgb_path else None
    name = os.path.basename(rgb_path) if rgb_path else ''
    note = ('<span style="color:#4fc3f7">Blue arrows</span> = DexPilot vec (original) | '
            '<span style="color:#ff7043">Orange arrows</span> = DexPilot vec (contact) | '
            'Colored dots = contact vertices | ★ = contact centroid')
    html = html_wrap(fig.to_html(full_html=False, include_plotlyjs='cdn'),
                     f'MANO Retargeting — frame {frame_num}', note,
                     b64, name, frame_num)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'[MANO 3D] -> {out_path}')


# ── matplotlib version ─────────────────────────────────────────────────────────

def _eq_axes(ax, pts, pad=0.01):
    x,y,z = pts[:,0],pts[:,1],pts[:,2]
    rng = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())/2+pad
    mid = np.array([(x.max()+x.min())/2,(y.max()+y.min())/2,(z.max()+z.min())/2])
    ax.set_xlim(mid[0]-rng,mid[0]+rng); ax.set_ylim(mid[1]-rng,mid[1]+rng)
    ax.set_zlim(mid[2]-rng,mid[2]+rng)


def _style(ax, title):
    ax.set_facecolor('#0d1117')
    ax.set_title(title, color='white', fontsize=8, pad=4)
    ax.tick_params(colors='#444', labelsize=4)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.10)


def draw_mano_panel(ax, sc, j_use, show_contact, vcol, elev, azim, title):
    vmp, faces = sc['vmp'], sc['faces']
    jo         = sc['jmp']
    cmask, lbl = sc['cmask'], sc['labels']
    centroids  = sc['centroids']
    idx        = sc['idx']

    _style(ax, title)
    poly = Poly3DCollection(vmp[faces], alpha=0.13, linewidth=0)
    poly.set_facecolor('#999'); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    ax.scatter(jo[:,0], jo[:,1], jo[:,2], c='#546e7a', s=10, depthshade=False)
    for a, b in _MANO_BONES:
        ax.plot([jo[a,0],jo[b,0]], [jo[a,1],jo[b,1]], [jo[a,2],jo[b,2]],
                c='#607d8b', lw=0.7, alpha=0.5)

    if show_contact and cmask.any():
        for fi in range(5):
            m = cmask & (lbl == fi)
            if m.any():
                cv = vmp[m]
                ax.scatter(cv[:,0],cv[:,1],cv[:,2], c=_FINGER_COLORS[fi],
                           s=6, alpha=0.7, depthshade=False, zorder=3)

    tips = jo[_FINGERTIP_KPT]
    ax.scatter(tips[:,0],tips[:,1],tips[:,2], c='#00bcd4', s=60, marker='o',
               depthshade=False, zorder=5, edgecolors='white', linewidths=0.6,
               label='orig fingertip')

    if show_contact:
        for fi, c in centroids.items():
            orig = jo[_FINGERTIP_KPT[fi]]
            ax.scatter(*c, c=_FINGER_COLORS[fi], s=140, marker='*',
                       depthshade=False, zorder=6, edgecolors='white', linewidths=0.4)
            d = c - orig
            ax.quiver(*orig, *d, color='white', alpha=0.7, lw=0.8, arrow_length_ratio=0.3)

    for i, (s, d) in enumerate(zip(idx[0], idx[1])):
        p0, p1 = j_use[s], j_use[d]
        ax.quiver(*p0, *(p1-p0), color=vcol, alpha=0.85, lw=1.3,
                  arrow_length_ratio=0.18, label='DexPilot vec' if i==0 else None)

    _eq_axes(ax, vmp); ax.view_init(elev=elev, azim=azim)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        if ll not in seen: seen[ll] = hh
    ax.legend(seen.values(), seen.keys(), loc='upper left', fontsize=5,
              framealpha=0.25, labelcolor='white', facecolor='#111133')


def save_mano_2d(scenes, frame_num, rgb_path, out_path):
    fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor('#1a1a2e')

    if rgb_path and os.path.exists(rgb_path):
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB))
        ax.set_title(f'RGB frame {frame_num}', color='white', fontsize=10)
        ax.axis('off')

    cfgs = [
        ('left',  False, '#4fc3f7', (3,4,7),  'Left — Original'),
        ('left',  True,  '#ff7043', (3,4,8),  'Left — Contact-adjusted'),
        ('right', False, '#4fc3f7', (3,4,10), 'Right — Original'),
        ('right', True,  '#ff7043', (3,4,11), 'Right — Contact-adjusted'),
    ]
    for hand, sc_flag, col, (nr,nc,idx), title in cfgs:
        ax = fig.add_subplot(nr, nc, idx, projection='3d')
        sc = scenes[hand]
        draw_mano_panel(ax, sc, sc['jc'] if sc_flag else sc['jmp'],
                        sc_flag, col, 15, -55, title)

    cr = [_FINGER_NAMES[i] for i in scenes['right']['centroids']]
    cl = [_FINGER_NAMES[i] for i in scenes['left']['centroids']]
    fig.suptitle(f'MANO Retargeting | frame {frame_num}\n'
                 f'Right: {", ".join(cr) or "none"}  |  Left: {", ".join(cl) or "none"}',
                 color='white', fontsize=11, y=1.01)
    plt.tight_layout(h_pad=2.0, w_pad=1.0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(); print(f'[MANO 2D] -> {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# ② MANO + xhand visualization
# ══════════════════════════════════════════════════════════════════════════════

def _xhand_traces(xfk, hand, col, lg):
    """Yellow xhand skeleton + DexPilot vectors (color=col)."""
    traces = []
    chains = xhand_chains(hand)
    all_pts = np.array([xfk[n] for c in chains for n in c if n in xfk])

    for ci, chain in enumerate(chains):
        for a, b in zip(chain[:-1], chain[1:]):
            if a not in xfk or b not in xfk: continue
            pa, pb = xfk[a], xfk[b]
            traces.append(go.Scatter3d(
                x=[pa[0],pb[0]], y=[pa[1],pb[1]], z=[pa[2],pb[2]],
                mode='lines', line=dict(color='#ffcc02', width=4),
                showlegend=(ci == 0 and a == chain[0]),
                name='xhand skeleton', legendgroup=f'{lg}_xskel'))
    if len(all_pts):
        traces.append(go.Scatter3d(
            x=all_pts[:,0], y=all_pts[:,1], z=all_pts[:,2], mode='markers',
            marker=dict(size=5, color='#ffcc02', line=dict(color='black', width=0.5)),
            name='xhand joints', showlegend=True, legendgroup=f'{lg}_xjts'))
    return traces, all_pts


def _xhand_vec_traces(xfk, opt, col, lg):
    traces = []
    for i, (on, tn) in enumerate(zip(opt.origin_link_names, opt.task_link_names)):
        if on not in xfk or tn not in xfk: continue
        traces += arrow(xfk[on], xfk[tn], col,
                        name='xhand DexPilot vec', show=(i == 0), lg=f'{lg}_xvec')
    return traces


def _xhand_panel_traces(sc, j_use, xfk, show_contact, mano_col, robot_col, lg):
    """MANO hand (no contact points) + centroids + DexPilot vecs + xhand."""
    vmp, faces = sc['vmp'], sc['faces']
    jo         = sc['jmp']
    centroids  = sc['centroids']
    idx        = sc['idx']
    opt        = sc['opt']
    hand       = sc['hand']
    traces     = []

    # MANO mesh
    traces.append(go.Mesh3d(
        x=vmp[:,0], y=vmp[:,1], z=vmp[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='lightgray', opacity=0.15, name='MANO mesh',
        showlegend=True, legendgroup=f'{lg}_mesh',
        lighting=dict(ambient=0.6, diffuse=0.8)))

    # MANO skeleton
    for a, b in _MANO_BONES:
        traces.append(go.Scatter3d(
            x=[jo[a,0],jo[b,0]], y=[jo[a,1],jo[b,1]], z=[jo[a,2],jo[b,2]],
            mode='lines', line=dict(color='#607d8b', width=2),
            showlegend=False, legendgroup=f'{lg}_skel'))
    traces.append(go.Scatter3d(
        x=jo[:,0],y=jo[:,1],z=jo[:,2], mode='markers',
        marker=dict(size=3.5, color='#78909c'),
        name='MANO joints', showlegend=True, legendgroup=f'{lg}_jts'))

    # original fingertips
    tips = jo[_FINGERTIP_KPT]
    traces.append(go.Scatter3d(
        x=tips[:,0],y=tips[:,1],z=tips[:,2], mode='markers',
        marker=dict(size=8, color='#00bcd4', symbol='circle',
                    line=dict(color='white', width=1)),
        name='Original fingertip', showlegend=True, legendgroup=f'{lg}_tip'))

    # contact centroids (no point cloud)
    if show_contact:
        for fi, c in centroids.items():
            orig = jo[_FINGERTIP_KPT[fi]]
            traces.append(go.Scatter3d(
                x=[c[0]],y=[c[1]],z=[c[2]], mode='markers',
                marker=dict(size=13, color=_FINGER_COLORS[fi], symbol='diamond',
                            line=dict(color='white', width=1)),
                name=f'{_FINGER_NAMES[fi]} centroid ★',
                showlegend=True, legendgroup=f'{lg}_cen{fi}'))
            traces += arrow(orig, c, 'white', name=f'{_FINGER_NAMES[fi]} shift',
                            show=True, lg=f'{lg}_sh{fi}')

    # MANO DexPilot vectors
    for i, (s, d) in enumerate(zip(idx[0], idx[1])):
        traces += arrow(j_use[s], j_use[d], mano_col,
                        name='MANO DexPilot vec', show=(i==0), lg=f'{lg}_mvec')

    # xhand skeleton
    xtr, _ = _xhand_traces(xfk, hand, robot_col, lg)
    traces += xtr

    # xhand DexPilot vectors
    traces += _xhand_vec_traces(xfk, opt, robot_col, lg)

    return traces


def save_xhand_3d(scenes, frame_num, rgb_path, out_path):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type':'scene'}]*2]*2,
        subplot_titles=[
            'Left — Original  (retarget_from_npz)',
            'Left — Contact-adjusted  (retarget_from_npz_contact)',
            'Right — Original  (retarget_from_npz)',
            'Right — Contact-adjusted  (retarget_from_npz_contact)',
        ],
        vertical_spacing=0.04, horizontal_spacing=0.02,
    )
    cfgs = [
        ('left',  False, '#4fc3f7', '#ffe066', 'lo', 1, 1),
        ('left',  True,  '#ff7043', '#ffa500', 'lc', 1, 2),
        ('right', False, '#4fc3f7', '#ffe066', 'ro', 2, 1),
        ('right', True,  '#ff7043', '#ffa500', 'rc', 2, 2),
    ]
    for hand, sc_flag, mc, rc, lg, r, c in cfgs:
        sc    = scenes[hand]
        j_use = sc['jc'] if sc_flag else sc['jmp']
        xfk   = sc['xhand_contact'] if sc_flag else sc['xhand_orig']
        for tr in _xhand_panel_traces(sc, j_use, xfk, sc_flag, mc, rc, lg):
            fig.add_trace(tr, row=r, col=c)

    for k in ['scene','scene2','scene3','scene4']:
        fig.update_layout(**{k: SCENE_CFG})

    cr = [_FINGER_NAMES[i] for i in scenes['right']['centroids']]
    cl = [_FINGER_NAMES[i] for i in scenes['left']['centroids']]
    fig.update_layout(
        paper_bgcolor='#1a1a2e', font=dict(color='white'),
        title=dict(
            text=(f'MANO + xhand Retargeting | frame {frame_num}<br>'
                  f'<sup>'
                  f'<span style="color:#4fc3f7">Blue</span>=MANO vec (original) | '
                  f'<span style="color:#ff7043">Orange</span>=MANO vec (contact) | '
                  f'<span style="color:#ffe066">Yellow skel</span>=xhand | '
                  f'<span style="color:#ffe066">Yellow arrow</span>=xhand vec (orig) | '
                  f'<span style="color:#ffa500">Orange arrow</span>=xhand vec (contact)<br>'
                  f'Right: {", ".join(cr) or "none"}  |  Left: {", ".join(cl) or "none"}'
                  f'</sup>'),
            font=dict(size=13, color='white'), x=0.5),
        legend=dict(bgcolor='rgba(20,20,40,0.7)', font=dict(size=9)),
        margin=dict(l=0,r=0,t=110,b=0), width=1600, height=900,
    )
    b64  = img_b64(rgb_path) if rgb_path else None
    name = os.path.basename(rgb_path) if rgb_path else ''
    note = ('<b>Left col</b>: retarget_from_npz.py result &nbsp;|&nbsp; '
            '<b>Right col</b>: retarget_from_npz_contact.py result &nbsp;|&nbsp; '
            'Yellow skeleton = xhand FK &nbsp;|&nbsp; ★ = contact centroid (adjusted fingertip)')
    html = html_wrap(fig.to_html(full_html=False, include_plotlyjs='cdn'),
                     f'MANO + xhand Retargeting — frame {frame_num}', note,
                     b64, name, frame_num)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'[xhand 3D] -> {out_path}')


def draw_xhand_panel(ax, sc, j_use, xfk, show_contact, mc, rc, elev, azim, title):
    vmp, faces = sc['vmp'], sc['faces']
    jo         = sc['jmp']
    centroids  = sc['centroids']
    idx        = sc['idx']
    opt        = sc['opt']
    hand       = sc['hand']

    _style(ax, title)
    poly = Poly3DCollection(vmp[faces], alpha=0.12, linewidth=0)
    poly.set_facecolor('#999'); poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    ax.scatter(jo[:,0],jo[:,1],jo[:,2], c='#546e7a', s=10, depthshade=False)
    for a, b in _MANO_BONES:
        ax.plot([jo[a,0],jo[b,0]],[jo[a,1],jo[b,1]],[jo[a,2],jo[b,2]],
                c='#607d8b', lw=0.7, alpha=0.5)

    tips = jo[_FINGERTIP_KPT]
    ax.scatter(tips[:,0],tips[:,1],tips[:,2], c='#00bcd4', s=55, marker='o',
               depthshade=False, zorder=5, edgecolors='white', linewidths=0.5,
               label='orig fingertip')

    if show_contact:
        for fi, c in centroids.items():
            orig = jo[_FINGERTIP_KPT[fi]]
            ax.scatter(*c, c=_FINGER_COLORS[fi], s=130, marker='*',
                       depthshade=False, zorder=6, edgecolors='white', linewidths=0.4,
                       label=f'{_FINGER_NAMES[fi]} centroid ★')
            d = c - orig
            ax.quiver(*orig, *d, color='white', alpha=0.7, lw=0.8, arrow_length_ratio=0.3)

    for i, (s, d) in enumerate(zip(idx[0], idx[1])):
        p0, p1 = j_use[s], j_use[d]
        ax.quiver(*p0, *(p1-p0), color=mc, alpha=0.85, lw=1.3,
                  arrow_length_ratio=0.18, label='MANO vec' if i==0 else None)

    # xhand skeleton
    chains  = xhand_chains(hand)
    all_xpt = np.array([xfk[n] for c in chains for n in c if n in xfk])
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            if a not in xfk or b not in xfk: continue
            pa, pb = xfk[a], xfk[b]
            ax.plot([pa[0],pb[0]],[pa[1],pb[1]],[pa[2],pb[2]],
                    c='#ffcc02', lw=2.0, alpha=0.95)
    if len(all_xpt):
        ax.scatter(all_xpt[:,0],all_xpt[:,1],all_xpt[:,2],
                   c='#ffcc02', s=20, depthshade=False, zorder=4,
                   edgecolors='black', linewidths=0.3, label='xhand joint')

    # xhand DexPilot vectors
    for i, (on, tn) in enumerate(zip(opt.origin_link_names, opt.task_link_names)):
        if on not in xfk or tn not in xfk: continue
        p0, p1 = xfk[on], xfk[tn]
        ax.quiver(*p0, *(p1-p0), color=rc, alpha=0.9, lw=1.3,
                  arrow_length_ratio=0.18, label='xhand vec' if i==0 else None)

    all_pts = np.vstack([vmp, all_xpt]) if len(all_xpt) else vmp
    _eq_axes(ax, all_pts); ax.view_init(elev=elev, azim=azim)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        if ll not in seen: seen[ll] = hh
    ax.legend(seen.values(), seen.keys(), loc='upper left', fontsize=5,
              framealpha=0.25, labelcolor='white', facecolor='#111133')


def save_xhand_2d(scenes, frame_num, rgb_path, out_path):
    fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor('#1a1a2e')

    if rgb_path and os.path.exists(rgb_path):
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB))
        ax.set_title(f'RGB frame {frame_num}', color='white', fontsize=10)
        ax.axis('off')

    cfgs = [
        ('left',  False, '#4fc3f7', '#ffe066', (3,4,7),  'Left — Original (retarget_from_npz)'),
        ('left',  True,  '#ff7043', '#ffa500', (3,4,8),  'Left — Contact-adj (retarget_from_npz_contact)'),
        ('right', False, '#4fc3f7', '#ffe066', (3,4,10), 'Right — Original (retarget_from_npz)'),
        ('right', True,  '#ff7043', '#ffa500', (3,4,11), 'Right — Contact-adj (retarget_from_npz_contact)'),
    ]
    for hand, sc_flag, mc, rc, (nr,nc,si), title in cfgs:
        ax  = fig.add_subplot(nr, nc, si, projection='3d')
        sc  = scenes[hand]
        xfk = sc['xhand_contact'] if sc_flag else sc['xhand_orig']
        draw_xhand_panel(ax, sc, sc['jc'] if sc_flag else sc['jmp'],
                         xfk, sc_flag, mc, rc, 15, -55, title)

    cr = [_FINGER_NAMES[i] for i in scenes['right']['centroids']]
    cl = [_FINGER_NAMES[i] for i in scenes['left']['centroids']]
    fig.suptitle(
        f'MANO + xhand | frame {frame_num}  '
        f'(blue/orange=MANO vec, yellow=xhand skel & vec)\n'
        f'Right: {", ".join(cr) or "none"}  |  Left: {", ".join(cl) or "none"}',
        color='white', fontsize=10, y=1.01)
    plt.tight_layout(h_pad=2.0, w_pad=1.0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(); print(f'[xhand 2D] -> {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--frame', type=int, default=10)
    ap.add_argument('--contact_dir', default=None,
                    help='Directory of per-frame contact npz (default: <episode>/contact)')
    ap.add_argument('--pkl_dir', default=None,
                    help='Directory containing qpos_xhand_*.pkl files '
                         '(default: same dir as --npz). '
                         'Needs qpos_xhand_{hand}.pkl AND qpos_xhand_contact_{hand}.pkl')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    out_dir     = args.out_dir or os.path.join(script_dir, 'vis')
    os.makedirs(out_dir, exist_ok=True)

    npz_dir     = os.path.dirname(os.path.abspath(args.npz))
    contact_dir = args.contact_dir or os.path.join(os.path.dirname(npz_dir), 'contact')
    pkl_dir     = args.pkl_dir or npz_dir
    rgb_img     = find_rgb(args.npz, args.frame)

    print(f'pkl_dir : {pkl_dir}')
    print(f'contact_dir: {contact_dir}')

    data  = np.load(args.npz)
    faces = load_mano_faces()
    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)

    scenes = {h: load_scene(data, faces, h, args.frame, contact_dir, pkl_dir)
              for h in ['left', 'right']}

    N = args.frame
    save_mano_2d(scenes, N, rgb_img, os.path.join(out_dir, f'frame{N}_mano_2d.png'))
    save_mano_3d(scenes, N, rgb_img, os.path.join(out_dir, f'frame{N}_mano_3d.html'))
    save_xhand_2d(scenes, N, rgb_img, os.path.join(out_dir, f'frame{N}_xhand_2d.png'))
    save_xhand_3d(scenes, N, rgb_img, os.path.join(out_dir, f'frame{N}_xhand_3d.html'))


if __name__ == '__main__':
    main()

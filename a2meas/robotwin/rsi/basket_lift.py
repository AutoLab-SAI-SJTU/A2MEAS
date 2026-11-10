"""One bounded inward/up lift after issued-source release and closed-arm stall.

FK describes our drive targets, never a measured basket
pose or proof that the hand holds anything. A current RGB basket association
is required before attempting the own-arm path.
"""
from collections import deque
import hashlib
import os
import numpy as np
from a2meas.robotwin.rsi.actor import Actor
from a2meas.robotwin.rsi.geometry import BASE_P,HEAD_C,HEAD_RCW,RightArmKinematics,camera_k,ee_world,project
from a2meas.robotwin.rsi.recovery import move_pose
from a2meas.robotwin.rsi.vision import MugRGBPerception


def new_memory():
    return dict(n=0,states=deque(maxlen=33),open_runs={6:0,13:0},closed_runs={6:0,13:0},first=None,source_released=False,done=False,rgb_checks=0)


def observe_commands(memory,state):
    memory['n']+=1;memory['state']=np.asarray(state,dtype=np.float32).copy();memory['states'].append(memory['state'])
    for grip in [6,13]:
        memory['open_runs'][grip]=memory['open_runs'][grip]+1 if state[grip]>.8 else 0
        memory['closed_runs'][grip]=memory['closed_runs'][grip]+1 if state[grip]<.2 else 0
    if memory['first'] is None:
        eligible=[g for g in [6,13] if memory['closed_runs'][g]>=16]
        if len(eligible)==1:memory['first']=eligible[0]
    if memory['first'] is not None and memory['open_runs'][memory['first']]>=16:
        memory['source_released']=True


def eligible_arm(memory,forecast,kin):
    if memory['done'] or memory['rgb_checks']>=2 or not 320<=memory['n']<=560 or not memory['source_released'] or len(memory['states'])<33:
        return None
    state=memory['state'];source=memory['first'];grip=13 if source==6 else 6
    if memory['closed_runs'][grip]<16 or state[source]<.8:return None
    side='left' if grip==6 else 'right';sl=slice(0,6) if side=='left' else slice(7,13)
    history=np.asarray(memory['states'])[:,sl]
    if np.max(np.linalg.norm(history-state[sl],axis=1))>=.1:return None
    positions=np.stack([ee_world(kin[side],q)[:3,3] for q in history]);now=positions[-1]
    if np.max(np.linalg.norm(positions-now,axis=1))>=.008:return None
    raw=np.asarray(forecast)
    if raw.shape!=(50,14) or not np.isfinite(raw).all():raise ValueError('Expected finite unchanged native50x14 forecast')
    if np.max(raw[:40,grip])>=.2:return None
    future=np.stack([ee_world(kin[side],q)[:3,3] for q in raw[:40,sl]])
    if np.max(np.linalg.norm(future-now,axis=1))>=.025 or np.max(future[:,2]-now[2])>=.015:return None
    return side


def inward_lift(kin,state,side,remaining):
    sl=slice(0,6) if side=='left' else slice(7,13);grip=6 if side=='left' else 13
    if state[grip]>=.2:raise ValueError('Only already-issued-closed hand supported')
    pose=ee_world(kin[side],state[sl]);point=pose[:3,3]
    inward=BASE_P[:2]-point[:2];length=float(np.linalg.norm(inward))
    if length<.1:raise ValueError('Ownwrist too close to base for stable inward direction')
    delta=np.r_[.06*inward/length,.06]
    rows=move_pose(kin[side],state,side,point+delta,R=pose[:3,:3],grip=0.,hold=8)
    if len(rows)+80>remaining or np.max(np.abs(rows[:,sl]-state[sl]))>=1.0:
        raise ValueError('Path exceeds frozen ownjoint or original700step reserve bound')
    other=list(range(7,14)) if side=='left' else list(range(7))
    if not np.array_equal(rows[:,other],np.broadcast_to(state[other],(len(rows),7))):
        raise ValueError('Other arm changed')
    return rows,delta


def unique_basket(detected,pixel,height,width):
    choices=[]
    for score,box in zip(detected['scores'],detected['boxes']):
        box=np.asarray(box);size=box[2:]-box[:2];area=float(np.prod(size));gap=float(np.linalg.norm(np.maximum(np.maximum(box[:2]-pixel,pixel-box[2:]),0)))
        if float(score)<.55 or not 800<area<.4*height*width or size[0]<20 or size[1]<25 or gap>15:continue
        choices.append(dict(box=box.tolist(),confidence=float(score),own_tool_bbox_gap_px=gap))
    return choices[0] if len(choices)==1 else None


class BasketInwardLift(Actor):
    def _load_model(self):
        super()._load_model();self.vision_model=self.vision(MugRGBPerception)
        urdf=os.environ['ROBOTWIN_ROOT']+'/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
        self.kin={s:RightArmKinematics(urdf,side=s) for s in ['left','right']}
        self.projection=camera_k()@np.c_[HEAD_RCW,-HEAD_RCW@HEAD_C];self._sessions={}


    async def on_episode_start(self,config,ctx):
        self._sessions[ctx.session_id]=new_memory()

    async def on_episode_end(self,result,ctx):
        self._sessions.pop(ctx.session_id,None)

    async def on_observation(self,obs,ctx):
        memory=self._sessions[ctx.session_id];observe_commands(memory,self._state_vector_from_obs(obs))
        memory['head']=self._image_from_obs_dict(obs['images'],self.image_name,0)
        return await super().on_observation(obs,ctx)

    def _normalize_result(self,result,ctx):
        sid=ctx.session_id;memory=self._sessions[sid];self._session_chunk_sizes[sid]=40
        native=super()._normalize_result(result,ctx)
        side=eligible_arm(memory,result['actions'],self.kin)
        if side is None:return native
        state=memory['state'];sl=slice(0,6) if side=='left' else slice(7,13)
        pose=ee_world(self.kin[side],state[sl]);pixel=project(self.projection,pose[:3,3]+.14*pose[:3,0])
        memory['rgb_checks']+=1
        # Segmentation is not a native-policy sampling step and must not consume
        # the policy RNG. No policy draw or episode gets reset here.
        with self._torch.random.fork_rng(devices=[self._torch.cuda.current_device()]):
            detected=self.vision_model.detect(memory['head'],'basket')
        basket=unique_basket(detected,pixel,*memory['head'].shape[:2])
        if basket is None:
            return native
        memory['done']=True
        try:rows,delta=inward_lift(self.kin,state,side,self.settings['max_steps']+1-memory['n'])
        except ValueError as exc:
            return native
        self._chunk_buffers.pop(sid,None);self._session_chunk_sizes[sid]=len(rows);self._try_serve_from_buffer(ctx)
        return {**native,'actions':rows,'joints':rows}

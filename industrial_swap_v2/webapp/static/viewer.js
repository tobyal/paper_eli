import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from 'spark';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';

const $=id=>document.getElementById(id);
const viewport=$('viewport');
let catalog,renderer,camera,scene,spark,orbit,activeObject,activePoints,backgroundMesh,backgroundKey;
let busy=false,fly=false,dragging=false,lastX=0,lastY=0,lastTime=0,frames=0,fpsTime=0;
const keys=new Set(),objects=new Map(),pointClouds=new Map();
const worldUp=new THREE.Vector3(),homeDirection=new THREE.Vector3();
const forward=new THREE.Vector3(),right=new THREE.Vector3(),movement=new THREE.Vector3();
const yaw=new THREE.Quaternion(),pitch=new THREE.Quaternion(),proposed=new THREE.Quaternion();
const names={object_m3:'原始 M3',object_m4:'M4 保存的设备',background_m1:'原始背景',background_m4:'优化背景'};
const resourceControls=['object-version','show-background','background-version','display-mode'];

function loading(title,detail=''){
  $('loading').hidden=false; $('loading-title').textContent=title; $('loading-detail').textContent=detail;
}
function failure(error){
  console.error(error); $('loading').hidden=true; $('error').hidden=false;
  $('error-detail').textContent=error.message||String(error); setBusy(false);
}
function setBusy(value){busy=value; for(const id of resourceControls) $(id).disabled=value;}
function updateTitle(){
  $('scene-title').textContent=`${names[$('object-version').value]} · ${$('show-background').checked?names[$('background-version').value]:'独立设备'}`;
  $('point-count').textContent=activeObject?.numSplats.toLocaleString('zh-CN')||'—';
  $('background-field').hidden=!$('show-background').checked;
}
async function loadSplat(key){
  const asset=catalog.resources[key]; if(!asset) throw Error('当前实验没有这份高斯模型');
  const mb=(asset.bytes/1024**2).toFixed(1);
  loading(`加载${names[key]}`,`${asset.count.toLocaleString()} 个高斯 · ${mb} MB`);
  const mesh=new SplatMesh({url:asset.url,fileName:`${key}.ply`,extSplats:true,lod:false,enableLod:false,
    onProgress:event=>{
      const loaded=(event.loaded/1024**2).toFixed(1);
      $('loading-detail').textContent=event.loaded>=asset.bytes?'正在解析高斯并准备渲染…':`${loaded} / ${mb} MB`;
    }});
  try {await mesh.initialized;} catch(error){mesh.dispose(); throw error;}
  if(mesh.numSplats!==asset.count){mesh.dispose(); throw Error('加载后的高斯数量与原始 PLY 不一致');}
  return mesh;
}
async function changeObject(){
  if(busy)return; setBusy(true); $('error').hidden=true;
  const key=$('object-version').value;
  try{
    if(!objects.has(key))objects.set(key,await loadSplat(key));
    activeObject?.removeFromParent(); activePoints?.removeFromParent(); activePoints=null;
    activeObject=objects.get(key); scene.add(activeObject);
    await displayMode(); updateTitle(); $('loading').hidden=true;
  }catch(error){failure(error);}finally{setBusy(false);}
}
async function changeBackground(){
  if(busy)return; setBusy(true); $('error').hidden=true;
  const key=$('background-version').value;
  try{
    if(!$('show-background').checked){if(backgroundMesh)backgroundMesh.visible=false;}
    else{
      if(backgroundMesh&&backgroundKey!==key){backgroundMesh.removeFromParent();backgroundMesh.dispose();backgroundMesh=null;}
      if(!backgroundMesh){backgroundMesh=await loadSplat(key);backgroundKey=key;scene.add(backgroundMesh);}
      backgroundMesh.visible=true;
    }
    $('loading').hidden=true; updateTitle();
  }catch(error){failure(error);}finally{setBusy(false);}
}
async function displayMode(){
  if(!activeObject)return;
  const points=$('display-mode').value==='points',key=$('object-version').value;
  if(points&&!pointClouds.has(key)){
    loading('构建设备中心点','只改变查看方式，不改动高斯文件');
    await new Promise(resolve=>requestAnimationFrame(resolve));
    const positions=new Float32Array(activeObject.numSplats*3),colors=new Float32Array(positions.length);
    activeObject.forEachSplat((index,center,scale,quaternion,opacity,color)=>{
      center.toArray(positions,index*3); color.toArray(colors,index*3);
    });
    const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:1.7,sizeAttenuation:false,vertexColors:true,toneMapped:false}));
    cloud.frustumCulled=false; pointClouds.set(key,cloud);
  }
  activeObject.visible=!points; activePoints?.removeFromParent(); activePoints=null;
  if(points){activePoints=pointClouds.get(key);scene.add(activePoints);}
  $('scene-hint').textContent=points?'中心点是位置诊断，不代表实体表面 · 高斯渲染可查看尺度与透明度':'左侧可切换背景，观察底座是否被遮挡 · F 聚焦设备';
  $('loading').hidden=true;
}
function placeCamera(target,direction,radius,fov=50){
  camera.up.copy(worldUp); camera.fov=fov; camera.position.copy(target).addScaledVector(direction,radius*3.3);
  camera.near=Math.max(.0001,catalog.geometry.radius*.001); camera.far=Math.max(100,catalog.geometry.radius*1000);
  orbit.target.copy(target); camera.lookAt(target); camera.updateProjectionMatrix(); orbit.update();
}
function focusObject(){if(!catalog)return;placeCamera(new THREE.Vector3().fromArray(catalog.geometry.center),homeDirection,catalog.geometry.radius);}
function focusBase(){
  if(!catalog)return; camera.getWorldDirection(forward).negate();
  placeCamera(new THREE.Vector3().fromArray(catalog.geometry.base_center),forward,catalog.geometry.base_radius);
}
function anchorView(){
  if(!catalog)return; camera.up.copy(worldUp); camera.position.fromArray(catalog.anchor.position);
  orbit.target.fromArray(catalog.anchor.target); camera.fov=catalog.anchor.fov;
  camera.lookAt(orbit.target); camera.updateProjectionMatrix(); orbit.update();
}
function navigation(){
  fly=$('navigation').value==='fly';keys.clear();dragging=false;orbit.enabled=!fly;
  if(!fly){camera.getWorldDirection(forward);orbit.target.copy(camera.position).addScaledVector(forward,catalog.geometry.radius*2);orbit.update();}
  $('speed-field').hidden=!fly;
  $('navigation-help').innerHTML=fly?'左键拖动转头 · W/S 前后 · A/D 左右<br>Q/E 上下 · Shift 加速 · 滚轮前后移动':'左键拖动旋转 · 右键拖动平移<br>滚轮缩放 · F 聚焦设备 · B 聚焦底座';
}
function turn(event){
  if(!fly||!dragging)return;
  const dx=event.clientX-lastX,dy=event.clientY-lastY;lastX=event.clientX;lastY=event.clientY;
  yaw.setFromAxisAngle(worldUp,-dx*.003);camera.quaternion.premultiply(yaw);
  right.set(1,0,0).applyQuaternion(camera.quaternion);
  pitch.setFromAxisAngle(right,-dy*.003);proposed.copy(camera.quaternion).premultiply(pitch);
  forward.set(0,0,-1).applyQuaternion(proposed);
  if(Math.abs(forward.dot(worldUp))<.995)camera.quaternion.copy(proposed);
}
function resize(){
  const w=viewport.clientWidth,h=viewport.clientHeight;
  renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();
}
function animate(ms){
  const dt=Math.min((ms-lastTime)/1000,.05);lastTime=ms;
  if(fly){
    camera.getWorldDirection(forward);right.set(1,0,0).applyQuaternion(camera.quaternion);movement.set(0,0,0);
    movement.addScaledVector(forward,Number(keys.has('KeyW'))-Number(keys.has('KeyS')));
    movement.addScaledVector(right,Number(keys.has('KeyD'))-Number(keys.has('KeyA')));
    movement.addScaledVector(worldUp,Number(keys.has('KeyE'))-Number(keys.has('KeyQ')));
    if(movement.lengthSq())camera.position.addScaledVector(movement.normalize(),dt*catalog.geometry.radius*Number($('speed').value)*(keys.has('ShiftLeft')||keys.has('ShiftRight')?3:1));
  }else orbit.update();
  renderer.render(scene,camera);frames++;
  if(ms-fpsTime>=1000){$('fps').textContent=`${Math.round(frames*1000/(ms-fpsTime))} FPS`;frames=0;fpsTime=ms;}
}
async function start(){
  const experiment=new URLSearchParams(location.search).get('experiment');
  const response=await fetch('/api/viewer/catalog'+(experiment?'?experiment_id='+encodeURIComponent(experiment):''));
  if(!response.ok)throw Error((await response.json()).detail||'无法读取实验');
  catalog=await response.json(); worldUp.fromArray(catalog.up);
  homeDirection.fromArray(catalog.anchor.position).sub(new THREE.Vector3().fromArray(catalog.geometry.center)).normalize();
  for(const id of ['object-version','background-version']){
    for(const option of [...$(id).options])if(!catalog.resources[option.value])option.remove();
  }
  renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor($('canvas-color').value,1);
  renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','可拖动的三维高斯视图');
  viewport.appendChild(renderer.domElement);
  camera=new THREE.PerspectiveCamera(50,1,.001,1000);scene=new THREE.Scene();
  spark=new SparkRenderer({renderer,accumExtSplats:true,enableLod:false,sortRadial:false,preBlurAmount:.3,blurAmount:0,maxStdDev:3});scene.add(spark);
  orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.dampingFactor=.12;
  orbit.minDistance=catalog.geometry.radius*.03;orbit.maxDistance=catalog.geometry.radius*200;orbit.zoomSpeed=.8;
  focusObject();resize();new ResizeObserver(resize).observe(viewport);
  renderer.domElement.addEventListener('pointerdown',event=>{
    renderer.domElement.focus();if(fly&&event.button===0){dragging=true;lastX=event.clientX;lastY=event.clientY;renderer.domElement.setPointerCapture(event.pointerId);}
  });
  renderer.domElement.addEventListener('pointermove',turn);
  renderer.domElement.addEventListener('pointerup',()=>dragging=false);renderer.domElement.addEventListener('pointercancel',()=>dragging=false);
  renderer.domElement.addEventListener('contextmenu',event=>event.preventDefault());
  renderer.domElement.addEventListener('wheel',event=>{if(fly){event.preventDefault();camera.getWorldDirection(forward);camera.position.addScaledVector(forward,-Math.sign(event.deltaY)*catalog.geometry.radius*.1);}},{passive:false});
  window.addEventListener('keydown',event=>{
    if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement?.tagName))return;
    if(event.code==='KeyF'){focusObject();event.preventDefault();}if(event.code==='KeyB'){focusBase();event.preventDefault();}
    if(fly&&['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(event.code)){keys.add(event.code);event.preventDefault();}
  });
  window.addEventListener('keyup',event=>keys.delete(event.code));window.addEventListener('blur',()=>{keys.clear();dragging=false;});
  document.addEventListener('visibilitychange',()=>keys.clear());
  renderer.setAnimationLoop(animate);
  $('object-version').onchange=changeObject;$('show-background').onchange=changeBackground;$('background-version').onchange=changeBackground;
  $('display-mode').onchange=()=>displayMode().catch(failure);$('navigation').onchange=navigation;
  $('focus-object').onclick=focusObject;$('focus-base').onclick=focusBase;$('anchor-view').onclick=anchorView;
  $('canvas-color').oninput=()=>renderer.setClearColor($('canvas-color').value,1);
  $('fullscreen').onclick=()=>{const action=document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();action.catch(failure);};
  await changeObject();
}
$('retry').onclick=()=>location.reload();
window.addEventListener('error',event=>{if(event.message)failure(new Error(event.message));});
window.addEventListener('unhandledrejection',event=>failure(event.reason));
start().catch(failure);

"""SQLite metadata and immutable, content-addressed asset storage."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid

from sqlalchemy import create_engine, text, event
from PIL import Image
import numpy as np
from plyfile import PlyData
import yaml

ROOT = Path(os.environ.get('SWAP_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
BASE = ROOT / 'industrial_swap_v2'
DATA = Path(os.environ.get('SWAP_WEB_DATA', BASE/'webapp/data')).resolve()
DATA.relative_to(BASE)
for directory in ('database', 'assets', 'experiments', 'tmp', 'service'):
    (DATA/directory).mkdir(parents=True, exist_ok=True)
engine = create_engine('sqlite:///'+str(DATA/'database/app.db'), connect_args={'timeout': 30})


@event.listens_for(engine, 'connect')
def pragmas(connection, _):
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('PRAGMA journal_mode=WAL')


def now(): return datetime.now(timezone.utc).isoformat()
def uid(): return uuid.uuid4().hex
def dump(value): return json.dumps(value, ensure_ascii=False, sort_keys=True)
def digest(value): return hashlib.sha256(dump(value).encode()).hexdigest()
def unpack(value): return json.loads(value or '{}')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


@contextmanager
def transaction():
    with engine.connect() as conn:
        conn.exec_driver_sql('BEGIN IMMEDIATE')
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def sql(conn, query, **params): return conn.execute(text(query), params)
def rows(query, **params):
    with engine.connect() as conn: return [dict(r) for r in sql(conn, query, **params).mappings()]
def one(query, **params):
    found=rows(query, **params)
    return found[0] if found else None


def init():
    statements = [
        '''CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, scene_sha TEXT NOT NULL,
            target_text TEXT NOT NULL, replacement_text TEXT NOT NULL, config TEXT NOT NULL,
            heads TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''',
        '''CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL,
            path TEXT NOT NULL, bytes INTEGER NOT NULL, source TEXT NOT NULL, model TEXT,
            prompt TEXT, metadata TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(sha256,kind))''',
        '''CREATE TABLE IF NOT EXISTS stage_runs (
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
            stage TEXT NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL,
            signature TEXT NOT NULL, inputs TEXT NOT NULL, parameters TEXT NOT NULL,
            config_path TEXT NOT NULL, folder TEXT NOT NULL, outputs TEXT NOT NULL DEFAULT '{}',
            log_path TEXT NOT NULL, reused_from TEXT, error TEXT, created_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, pid INTEGER, attempt INTEGER NOT NULL DEFAULT 1,
            request_id TEXT NOT NULL UNIQUE)''',
        '''CREATE TABLE IF NOT EXISTS experiment_assets (
            experiment_id TEXT NOT NULL REFERENCES experiments(id), asset_id TEXT NOT NULL REFERENCES assets(id),
            role TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            PRIMARY KEY(experiment_id,asset_id,role))''',
        '''CREATE TABLE IF NOT EXISTS generation_records (
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
            run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(id), model TEXT, prompt TEXT,
            input_assets TEXT NOT NULL, output_assets TEXT NOT NULL DEFAULT '[]',
            external_api INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, seconds REAL,
            metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)''',
        '''CREATE TABLE IF NOT EXISTS asset_events (
            id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id), experiment_id TEXT,
            source TEXT NOT NULL, name TEXT, model TEXT, prompt TEXT, metadata TEXT NOT NULL, created_at TEXT NOT NULL)''',
        'CREATE INDEX IF NOT EXISTS runs_cache ON stage_runs(signature,status)',
    ]
    with transaction() as conn:
        for statement in statements: conn.exec_driver_sql(statement)


def config(): return yaml.safe_load((BASE/'configs/default.yaml').read_text(encoding='utf-8'))


def attach(conn, experiment, asset, role, selected=False):
    if selected:
        sql(conn,'UPDATE experiment_assets SET selected=0 WHERE experiment_id=:e AND role=:r',e=experiment,r=role)
    sql(conn,'''INSERT INTO experiment_assets(experiment_id,asset_id,role,selected,created_at)
        VALUES(:e,:a,:r,:s,:t) ON CONFLICT(experiment_id,asset_id,role) DO UPDATE SET selected=MAX(selected,:s)''',
        e=experiment,a=asset,r=role,s=int(selected),t=now())


def validate_file(path, kind):
    path=Path(path); meta={}
    if kind in ('edited_scene','object_rgba','mask','depth','preview'):
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if min(im.size)<16 or max(im.size)>8192: raise ValueError('图片尺寸必须为 16–8192 像素')
            meta.update(width=im.width,height=im.height)
            if kind=='object_rgba':
                if 'A' not in im.getbands(): raise ValueError('RGBA 素材必须包含透明通道')
                alpha=np.asarray(im.getchannel('A'))
                if not (alpha>127).any() or not (alpha<128).any():
                    raise ValueError('RGBA 需要同时包含有效前景和透明背景')
    elif kind in ('gaussian_local','gaussian_world'):
        vertices=PlyData.read(str(path))['vertex'].data
        required=['x','y','z','nx','ny','nz','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        if set(required)!=set(vertices.dtype.names) or not 1<=len(vertices)<=10_000_000:
            raise ValueError('需要 Graphdeco SH3 Gaussian PLY，普通点云请选择 Mesh 类型')
        if any(not np.isfinite(vertices[k]).all() for k in required): raise ValueError('PLY 包含非有限数值')
        if (sum(vertices[f'rot_{i}']**2 for i in range(4))<1e-12).any(): raise ValueError('PLY 存在无效四元数')
        if any((np.abs(vertices[f'scale_{i}'])>30).any() for i in range(3)): raise ValueError('PLY 高斯尺度超出可渲染范围')
        meta['points']=len(vertices)
    elif kind=='mesh':
        suffix=path.suffix.lower()
        if suffix not in ('.glb','.obj','.stl','.ply'): raise ValueError('支持 GLB、独立 OBJ、STL 和 Mesh PLY')
        if suffix=='.glb':
            with path.open('rb') as f:
                header=f.read(12)
                if len(header)!=12: raise ValueError('GLB 文件头不完整')
                magic,version,length=struct.unpack('<4sII',header)
                if magic!=b'glTF' or version!=2 or length!=path.stat().st_size: raise ValueError('GLB 文件头无效')
                chunk=f.read(8)
                if len(chunk)!=8: raise ValueError('GLB 缺少 JSON 块')
                size,kind_code=struct.unpack('<II',chunk)
                if kind_code!=0x4e4f534a or size>32*1024**2: raise ValueError('GLB JSON 块无效')
                scene=json.loads(f.read(size))
                for resource in scene.get('buffers',[])+scene.get('images',[]):
                    uri=resource.get('uri','')
                    if uri and not uri.startswith('data:'): raise ValueError('请上传自包含 GLB，不允许外部文件或网址引用')
        if suffix=='.obj':
            with path.open(errors='replace') as f:
                if any(line.lstrip().startswith('mtllib') for line in f):
                    raise ValueError('带外部 MTL/纹理的 OBJ 请先打包导出为 GLB')
        if suffix=='.ply':
            data=PlyData.read(str(path))
            if 'face' not in data or len(data['face'].data)==0:
                raise ValueError('Mesh PLY 必须包含面；高斯点云请选择 Gaussian PLY 类型')
    elif kind not in ('checkpoint','video','metadata'): raise ValueError('不支持的素材类型')
    return meta


def register(path, kind, name=None, source='existing_experiment', model=None, prompt=None, metadata=None,
             experiment=None, role=None, trusted=False):
    path=Path(path).resolve()
    # Inputs are uploads in DATA/tmp or known project outputs, never arbitrary server paths.
    if not any(path.is_relative_to(p) for p in (DATA, BASE/'outputs')): raise ValueError('素材路径不在允许目录')
    info={} if trusted else validate_file(path,kind)
    info.update(metadata or {})
    checksum=sha(path)
    folder=DATA/'assets'/checksum[:2]; folder.mkdir(exist_ok=True)
    destination=folder/(checksum+path.suffix.lower())
    if not destination.exists():
        temporary=folder/(checksum+'.'+uid()+'.tmp')
        try: os.link(path,temporary)
        except OSError: shutil.copy2(path,temporary)
        os.replace(temporary,destination)
    with transaction() as conn:
        old=sql(conn,'SELECT * FROM assets WHERE sha256=:h AND kind=:k',h=checksum,k=kind).mappings().first()
        identifier=old['id'] if old else uid()
        if not old:
            sql(conn,'''INSERT INTO assets VALUES(:id,:name,:kind,:sha,:path,:bytes,:source,:model,:prompt,:meta,:time)''',
                id=identifier,name=name or path.name,kind=kind,sha=checksum,path=str(destination),bytes=path.stat().st_size,
                source=source,model=model,prompt=prompt,meta=dump(info),time=now())
        if experiment: attach(conn,experiment,identifier,role or kind)
        sql(conn,'INSERT INTO asset_events VALUES(:id,:a,:e,:source,:name,:model,:prompt,:meta,:t)',
            id=uid(),a=identifier,e=experiment,source=source,name=name or path.name,model=model,prompt=prompt,meta=dump(info),t=now())
    return identifier


def asset(identifier):
    value=one('SELECT * FROM assets WHERE id=:id',id=identifier)
    if not value: raise ValueError('素材不存在')
    return value


def asset_path(identifier, verify=False):
    value=asset(identifier); path=Path(value['path']).resolve()
    path.relative_to(DATA/'assets')
    if not path.is_file() or path.stat().st_size!=value['bytes']: raise ValueError('素材文件缺失或大小变化')
    if verify and sha(path)!=value['sha256']: raise ValueError('素材哈希校验失败')
    return path


def public_asset(value):
    value=dict(value); value['storage_path']=str(Path(value.pop('path')).relative_to(DATA)); value['metadata']=unpack(value['metadata'])
    value['url']='/api/assets/'+value['id']+'/file'
    value['experiments']=rows('''SELECT e.id,e.name,ea.role,ea.selected FROM experiment_assets ea
        JOIN experiments e ON e.id=ea.experiment_id WHERE ea.asset_id=:a''',a=value['id'])
    value['events']=rows('SELECT source,name,model,experiment_id,created_at FROM asset_events WHERE asset_id=:a ORDER BY created_at DESC LIMIT 30',a=value['id'])
    return value

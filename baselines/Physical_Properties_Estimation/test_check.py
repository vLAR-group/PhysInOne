"""Offline regression tests: python -m unittest discover -s . -p test_check.py -v."""
import ast
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

from PIL import Image

SPEC = importlib.util.spec_from_file_location("ppe_check", Path(__file__).with_name("check.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
SCENE = "ElasticFall__bg851__fixture_trajectory"
SAMPLES = {
    "ELASTIC": {
        "E": 292756.78125,
        "nu": 0.4689558148384094
    },
    "SAND": {
        "friction_angle": 42.60722351074219
    },
    "NEWTONIAN": {
        "mu": 88.37726593017578,
        "kappa": 5461.81982421875
    },
    "NON_NEWTONIAN": {
        "mu": 11176.0625,
        "kappa": 434403872.0,
        "yield": 12337.998046875,
        "visc": 9.200647354125977
    },
    "PLASTICINE": {
        "E": 6551366.0,
        "nu": 0.15575304627418518,
        "yield": 4540.87744140625
    }
}


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ppe-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = dict(material="ELASTIC",
            training_cameras=["CineCamera_2", "CineCamera_7"],
            novel_cameras=["CineCamera_0", "CineCamera_4"],
            frame_stems=["0000", "0001", "0002"], image_size=[16,16])
        self.manifest = dict(schema_version=1, scenes={SCENE:self.entry})
        self.material = {"ELASTIC":copy.deepcopy(SAMPLES["ELASTIC"])}
        self.velocity = dict(vx=0,vy=-0.1,vz=0.2)

    def members(self, ext=".png", flat=False, mode="RGB"):
        result = {"materials_configs.json":json.dumps(self.material).encode(),
                  "initvelocity.json":json.dumps(self.velocity).encode()}
        groups = {"novel":self.entry["novel_cameras"],
                  "reinit":self.entry["training_cameras"]+self.entry["novel_cameras"]}
        for group,cameras in groups.items():
            for camera in cameras:
                for i,stem in enumerate(self.entry["frame_stems"]):
                    im=Image.new("RGB",(16,16),(255,255,255) if flat else (30+i*40,60,100)).convert(mode)
                    stream=io.BytesIO()
                    im.save(stream,format=CHECK.EXTENSIONS[ext])
                    result[f"{group}/{camera}/rgb/{stem}{ext}"]=stream.getvalue()
        return result

    def write_zip(self,members,name=None):
        path=self.root/(name or SCENE+".zip")
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,"w") as archive:
            for name,raw in members.items():archive.writestr(name,raw)
        return path

    def check(self,members):
        return CHECK.validate_zip(self.write_zip(members),self.entry)

    def test_complete_png_output(self):
        result=self.check(self.members())
        self.assertEqual(result["errors"],[])
        self.assertEqual(result["warnings"],[])
        self.assertEqual(result["images"],18)
        self.assertTrue(result["parameters_valid"])

    def test_jpeg_and_mixed_image_extensions(self):
        for ext in (".jpg",".jpeg"):
            self.assertEqual(self.check(self.members(ext=ext))["errors"],[])
        members=self.members()
        key="novel/CineCamera_0/rgb/0000"
        del members[key+".png"]
        members[key+".jpg"]=self.members(ext=".jpg")[key+".jpg"]
        self.assertEqual(self.check(members)["errors"],[])

    def test_payload_wrappers_and_explicit_directories(self):
        for wrapper in ("property_estimation/",SCENE+"/",SCENE+"/property_estimation/"):
            members={wrapper+key:value for key,value in self.members().items()}
            members[wrapper]=b""
            self.assertEqual(self.check(members)["errors"],[])
        members=self.members()
        members["novel/"]=b""
        members["novel/CineCamera_0/rgb/"]=b""
        self.assertEqual(self.check(members)["errors"],[])

    def test_missing_novel_or_reinit_group(self):
        for group in ("novel/","reinit/"):
            members={k:v for k,v in self.members().items() if not k.startswith(group)}
            self.assertTrue(any("Missing" in e for e in self.check(members)["errors"]))

    def test_reinit_requires_training_and_novel_cameras(self):
        for prefix in ("reinit/CineCamera_2/","reinit/CineCamera_4/"):
            members={k:v for k,v in self.members().items() if not k.startswith(prefix)}
            self.assertTrue(any(prefix.rstrip('/') in e for e in self.check(members)["errors"]))

    def test_wrong_or_empty_camera_directories(self):
        for name in ("novel/CineCamera_2/","reinit/CineCamera_99/","novel/CineCamera_Moving/"):
            members=self.members();members[name]=b""
            self.assertTrue(any("Unexpected" in e for e in self.check(members)["errors"]))
        members={k:v for k,v in self.members().items() if not k.startswith("novel/CineCamera_0/")}
        members["novel/CineCamera_0/"]=b""
        self.assertTrue(self.check(members)["errors"])

    def test_frame_names_are_exact(self):
        key="novel/CineCamera_0/rgb/0000.png"
        for wrong in ("0.png","0003.png","0075.png"):
            members=self.members()
            members["novel/CineCamera_0/rgb/"+wrong]=members.pop(key)
            result=self.check(members)
            self.assertTrue(any("Missing" in e for e in result["errors"]))
            self.assertTrue(any("Unexpected" in e for e in result["errors"]))

    def test_duplicate_frame_stems(self):
        members=self.members()
        key="novel/CineCamera_0/rgb/0000.jpg"
        members[key]=self.members(ext=".jpg")[key]
        self.assertTrue(any("Duplicate frame" in e for e in self.check(members)["errors"]))

    def test_unexpected_files_are_rejected(self):
        for filename in ("weights.pt","0-pred.json","camera_mapping.json","novel/CineCamera_0/video.mp4",
                         "resimulate/CineCamera_0/rgb/0000.png"):
            members=self.members();members[filename]=b"extra"
            self.assertTrue(any("Unexpected file" in e for e in self.check(members)["errors"]))

    def test_both_parameter_files_are_required(self):
        for filename in CHECK.JSON_FILES:
            members=self.members();del members[filename]
            self.assertTrue(any("Missing parameter file" in e for e in self.check(members)["errors"]))

    def test_all_material_schemas(self):
        for material,values in SAMPLES.items():
            self.entry["material"]=material
            self.material={material:copy.deepcopy(values)}
            self.assertEqual(self.check(self.members())["errors"],[])
        self.entry["material"]="NON_NEWTONIAN"
        self.material={"NON_NEWTONIAN":dict(SAMPLES["NON_NEWTONIAN"],plastic_pow=1)}
        self.assertTrue(self.check(self.members())["errors"])

    def test_missing_extra_or_wrong_material(self):
        for payload in ({"ELASTIC":{"E":100}}, {"ELASTIC":dict(SAMPLES["ELASTIC"],rho=1000)},
                        {"SAND":SAMPLES["SAND"]}, {}, {"ELASTIC":[]}):
            self.material=payload
            self.assertTrue(self.check(self.members())["errors"])

    def test_raw_alpha_is_not_a_friction_angle(self):
        self.entry["material"]="SAND"
        self.material={"SAND":dict(E=1000,nu=.3,friction_alpha=.1)}
        errors=self.check(self.members())["errors"]
        self.assertTrue(any("friction_angle" in e for e in errors))
        self.assertTrue(any("friction_alpha" in e for e in errors))

    def test_physical_parameter_ranges(self):
        for material,key,value in (("ELASTIC","E",0),("ELASTIC","nu",.5),("ELASTIC","nu",-1),
            ("NEWTONIAN","mu",-1),("NEWTONIAN","kappa",0),("PLASTICINE","yield",-1),
            ("NON_NEWTONIAN","visc",-1),("SAND","friction_angle",-10),
            ("SAND","friction_angle",90)):
            self.entry["material"]=material
            self.material={material:dict(SAMPLES[material],**{key:value})}
            self.assertTrue(self.check(self.members())["errors"],(material,key,value))
        self.entry["material"]="SAND"
        self.material={"SAND":dict(SAMPLES["SAND"],friction_angle=0)}
        self.assertEqual(self.check(self.members())["errors"],[])

    def test_previous_submission_names_are_rejected(self):
        cases=[('SAND',{'GRANULAR':{'E':1000,'nu':.3,'friction_angle':20}}),
               ('SAND',{'SAND':{'E':1000,'nu':.3,'friction_angle':20}}),
               ('NON_NEWTONIAN',{'NON_NEWTONIAN':{'mu':1,'kappa':2,'yield_stress':3,'plastic_viscosity':4}}),
               ('PLASTICINE',{'PLASTICINE':{'E':1000,'nu':.25,'yield_stress':3}})]
        for material,payload in cases:
            self.entry['material']=material
            self.material=payload
            with self.subTest(material=material,payload=payload):
                self.assertTrue(self.check(self.members())['errors'])

    def test_invalid_numeric_parameter_types(self):
        for value in (True,"1000",[1000],None,float("nan"),float("inf")):
            self.material={"ELASTIC":dict(E=value,nu=.25)}
            self.assertTrue(self.check(self.members())["errors"])

    def test_invalid_velocity_shape_and_numbers(self):
        for value in ([0,0,0],{"vx":0,"vy":0},{"vx":0,"vy":0,"vz":0,"omega":0},
                      {"vx":False,"vy":0,"vz":0},{"vx":float("inf"),"vy":0,"vz":0}):
            self.velocity=value
            self.assertTrue(self.check(self.members())["errors"])

    def test_duplicate_and_invalid_json(self):
        for raw in (b'{"vx":0,"vx":1,"vy":0,"vz":0}',b'not JSON',b'\xff',b'',
                    b'{"vx":NaN,"vy":0,"vz":0}'):
            members=self.members();members["initvelocity.json"]=raw
            self.assertTrue(self.check(members)["errors"])

    def test_image_decode_extension_size_and_mode(self):
        key="novel/CineCamera_0/rgb/0000.jpg"
        for raw in (b"",b"not an image",self.members()[key.replace('.jpg','.png')]):
            members=self.members(ext=".jpg");members[key]=raw
            self.assertTrue(self.check(members)["errors"])
        self.assertTrue(self.check(self.members(mode="L"))["errors"])
        self.assertTrue(self.check(self.members(mode="RGBA"))["errors"])
        self.entry["image_size"]=[32,32]
        self.assertTrue(self.check(self.members())["errors"])

    def test_truncated_jpeg_and_opencv_failure(self):
        members=self.members(ext=".jpg")
        key="novel/CineCamera_0/rgb/0000.jpg"
        members[key]=members[key][:-20]
        self.assertTrue(self.check(members)["errors"])
        _,_,cv2=CHECK.dependencies()
        with patch.object(cv2,"imdecode",return_value=None):
            self.assertTrue(self.check(self.members())["errors"])

    def test_90_frames_two_novel_and_15_reinit_views(self):
        data=CHECK.load_manifest(CHECK.DEFAULT_MANIFEST)
        self.entry=copy.deepcopy(next(iter(data['scenes'].values())))
        self.entry['image_size']=[16,16]
        self.material={self.entry['material']:copy.deepcopy(SAMPLES[self.entry['material']])}
        result=self.check(self.members(ext='.jpg'))
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['images'],90*(2+15))
        self.assertEqual(len(result['camera_summary']),17)

    def test_oversized_decoded_image_is_reported(self):
        members=self.members()
        with patch.object(Image,'open',side_effect=Image.DecompressionBombError('too large')):
            result=self.check(members)
        self.assertTrue(result['errors'])
        self.assertEqual(result['images'],0)

    def test_content_warnings(self):
        result=self.check(self.members(flat=True))
        self.assertEqual(result["errors"],[])
        self.assertTrue(any("white" in w for w in result["warnings"]))
        self.assertTrue(any("identical" in w for w in result["warnings"]))

    def test_unsafe_paths_and_symlinks(self):
        for name in ("../escape","/absolute","C:\\escape","novel//extra","novel/./extra"):
            members=self.members();members[name]=b"unsafe"
            self.assertTrue(self.check(members)["errors"])
        path=self.write_zip(self.members())
        with zipfile.ZipFile(path,"a") as archive:
            info=zipfile.ZipInfo("link");info.create_system=3;info.external_attr=0o120777<<16
            archive.writestr(info,"outside")
        self.assertTrue(CHECK.validate_zip(path,self.entry)["errors"])

    def test_duplicate_members_and_limits(self):
        path=self.write_zip(self.members())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(path,"a") as archive:archive.writestr("initvelocity.json",b"duplicate")
        self.assertTrue(CHECK.validate_zip(path,self.entry)["errors"])
        for flag,size in ((1,0),(0,CHECK.MAX_MEMBER_BYTES+1)):
            info=zipfile.ZipInfo("image.jpg");info.flag_bits=flag;info.file_size=size
            with self.assertRaises(ValueError):CHECK.member_path(info)
        with patch.object(CHECK,"MAX_JSON_BYTES",1):
            self.assertTrue(self.check(self.members())["errors"])

    def test_crc_failure_and_invalid_deflate(self):
        members=self.members();path=self.write_zip(members)
        raw=bytearray(path.read_bytes())
        image=next(value for key,value in members.items() if key.endswith('.png'))
        raw[raw.index(image)+40]^=1
        path.write_bytes(raw)
        self.assertTrue(CHECK.validate_zip(path,self.entry)["errors"])
        path=self.write_zip(members)
        with patch.object(zipfile.ZipFile,"testzip",side_effect=zlib.error("bad stream")):
            self.assertTrue(CHECK.validate_zip(path,self.entry)["errors"])

    def test_inventory_missing_extra_duplicate(self):
        self.assertEqual(CHECK.check_submission(self.root,self.manifest)["summary"]["missing"],1)
        self.write_zip(self.members())
        self.write_zip(self.members(),"extra.zip")
        self.write_zip(self.members(),"nested/"+SCENE+".zip")
        report=CHECK.check_submission(self.root,self.manifest)
        self.assertEqual(report["summary"]["extra"],1)
        self.assertTrue(report["scenes"][SCENE]["errors"])

    def test_manifest_validation_and_bundled_scope(self):
        path=self.root/"manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(CHECK.load_manifest(path),self.manifest)
        for key,value in (("material","UNKNOWN"),("material",[]),("novel_cameras",["CineCamera_2"]),
                          ("frame_stems",["0"]),("image_size",[0,16])):
            bad=copy.deepcopy(self.manifest);bad["scenes"][SCENE][key]=value
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):CHECK.load_manifest(path)
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):CHECK.load_manifest(path)
        data=CHECK.load_manifest(CHECK.DEFAULT_MANIFEST)
        self.assertEqual(len(data["scenes"]),60)
        self.assertEqual(set(data),{"schema_version","defaults","scenes"})
        self.assertTrue(all(set(e)=={"material","training_cameras","novel_cameras",
                                    "frame_stems","image_size"} for e in data["scenes"].values()))
        for entry in data["scenes"].values():
            self.assertEqual((len(entry["training_cameras"]),len(entry["novel_cameras"])),(13,2))
            self.assertEqual(entry["frame_stems"],[f'{i:04d}' for i in range(90)])
            self.assertEqual(entry["image_size"],[1120,1120])

    def test_shared_manifest_defaults_and_scene_overrides(self):
        compact=dict(schema_version=1,defaults={key:self.entry[key] for key in ('frame_stems','image_size')},
            scenes={SCENE:{key:self.entry[key] for key in ('material','training_cameras','novel_cameras')}})
        compact['scenes']['another_trajectory']=dict(compact['scenes'][SCENE],
            image_size=[32,32],frame_stems=['0005'])
        path=self.root/'compact_manifest.json'
        path.write_text(json.dumps(compact),encoding='utf-8')
        before=path.read_bytes()
        data=CHECK.load_manifest(path)
        self.assertEqual(data['scenes'][SCENE],self.entry)
        self.assertEqual(data['scenes']['another_trajectory']['image_size'],[32,32])
        self.assertEqual(data['scenes']['another_trajectory']['frame_stems'],['0005'])
        self.assertEqual(path.read_bytes(),before)
        for defaults in (None,{'fps':150},{'image_size':[0,16]}):
            invalid=copy.deepcopy(compact);invalid['defaults']=defaults
            path.write_text(json.dumps(invalid),encoding='utf-8')
            with self.assertRaises(ValueError):CHECK.load_manifest(path)

    def test_all_four_method_parameter_exports_are_compatible(self):
        _,np,_=CHECK.dependencies()
        methods=Path(__file__).parent/'Physical_Properties_Estimation'
        codes={'ELASTIC':10,'NEWTONIAN':11,'NON_NEWTONIAN':12,'PLASTICINE':12,'SAND':13}
        for method in ('PAC-NeRF','PAC-NeRF-complex','gic','gic-complex'):
            for label,values in SAMPLES.items():
                directory=self.root/method/label
                directory.mkdir(parents=True)
                native=dict(values)
                if 'yield' in native:native['yield_stress']=native.pop('yield')
                if 'visc' in native:native['plastic_viscosity']=native.pop('visc')
                if label=='SAND':native.update(E=1000000,nu=.3)
                if label=='NON_NEWTONIAN':native['plastic_pow']=1
                if method.startswith('gic'):
                    path=methods/method/'inference.py'
                    tree=ast.parse(path.read_text(encoding='utf-8'))
                    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export_parameters')
                    def save_json(path,payload):
                        Path(path).write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
                    namespace=dict(save_json=save_json,Path=Path)
                    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
                    params=dict(native,material=codes[label])
                    if label=='SAND':params['friction_alpha']=params.pop('friction_angle')
                    physics=dict(mat_params=params,vel=[0,-.1,.2])
                    before=copy.deepcopy(physics)
                    namespace['export_parameters'](physics,directory)
                    self.assertEqual(physics,before)
                else:
                    path=methods/method/'model_params.py'
                    tree=ast.parse(path.read_text(encoding='utf-8'))
                    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_estimated_model_params')
                    decoded=dict(native)
                    export_module=ModuleType('property_export')
                    export_module.property_output_dir=lambda *args:str(directory)
                    namespace=dict(os=os,json=json,np=np,
                        MPMSimulator=NS(elasticity=10,viscous_fluid=11,von_mises=12,drucker_prager=13),
                        collect_estimated_model_params=lambda p:dict(material_parameters=decoded,
                            object_initial_velocity=[0,-.1,.2]))
                    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
                    with patch.dict(sys.modules,{'property_export':export_module}), redirect_stdout(io.StringIO()):
                        namespace['save_estimated_model_params'](NS(material=codes[label]),{},NS())
                self.entry['material']=label
                self.material=json.loads((directory/'materials_configs.json').read_text())
                self.velocity=json.loads((directory/'initvelocity.json').read_text())
                with self.subTest(method=method,material=label):
                    self.assertEqual(self.material,{label:values})
                    self.assertEqual(self.velocity,dict(vx=0,vy=-.1,vz=.2))
                    self.assertEqual(self.check(self.members())['errors'],[])

    def test_cli_exit_codes_reports_and_read_only_archives(self):
        archive=self.write_zip(self.members(flat=True));before=archive.read_bytes()
        manifest=self.root/"manifest.json";manifest.write_text(json.dumps(self.manifest))
        command=[sys.executable,'-B',str(Path(CHECK.__file__)),str(self.root),"--manifest",str(manifest)]
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
        self.assertEqual(subprocess.run(command+["--fail-on-warning"],capture_output=True).returncode,1)
        report=self.root/"report.json"
        self.assertEqual(subprocess.run(command+["--report",str(report)],capture_output=True).returncode,0)
        self.assertEqual(subprocess.run(command+["--report",str(report)],capture_output=True).returncode,2)
        self.assertEqual(archive.read_bytes(),before)
        archive.unlink()
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,1)


if __name__ == "__main__":
    unittest.main()

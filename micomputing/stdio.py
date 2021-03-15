#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

# builtins
# nothing yet

# checked
import torch
import torchplus as tp

try:
    import nibabel as nib
    import pydicom as dcm
except ImportError:
    raise ImportError("Inout system of 'pyctlib.mic' cannot be used without dependencies 'nibabel' and 'pydicom'.")

# numpy & pyctlib should exists because of torchplus
import numpy as np
from pyctlib import path, shell

def toU(dt):
    if np.dtype(dt).kind == np.dtype(np.int).kind: return np.dtype('uint%d'%(8*dt.itemsize))
    else: return dt
    
def toI(dt):
    if np.dtype(dt).kind == np.dtype(np.uint).kind: return np.dtype('int%d'%(8*dt.itemsize))
    else: return dt

def orderedValue(d):
    return [d[k] for k in sorted(d.keys())]

def orn2quatern(r11, r21, r31, r12, r22, r32):
    eps = 1e-6
    r11, r12, r21, r22 = -r11, -r12, -r21, -r22
    A, B, C = 1, -(r11+r22+2) / 2, (r11+1) * (r22+1) / 4 - ((r12+r21) / 4) ** 2
    quatern_a_sqs = [(-B + np.sqrt(B*B-4*A*C)) / (2*A), (-B - np.sqrt(B*B-4*A*C)) / (2*A)]
    quatern_a = [np.sqrt(np.clip(x, 0, 1)) for x in quatern_a_sqs if -eps <= x <= 1 + eps]
    if len(quatern_a) <= 0: raise TypeError("Invalid orientation")
    if len(quatern_a) > 1:
        r13 = r21 * r32 - r31 * r22
        r23 = - r11 * r32 + r31 * r12
        r33 = r11 * r22 - r21 * r12
        r = np.sqrt(r13 ** 2 + r23 ** 2 + r33 ** 2)
        t = np.abs(r33) / r
        quatern_a = [x for x in quatern_a if np.abs(np.abs(4 * x * x - 1 - r11 - r22) - t) < eps]
        if len(quatern_a) <= 0 or len(set(quatern_a)) > 1: raise TypeError("Invalid orientation")
    qa = np.clip(quatern_a[0], 0, 1)
    if np.abs(qa) > eps:
        qd = (r21 - r12) / qa / 4
        qc = (qd * r32 - qa * r31) / (qd ** 2 + qa ** 2) / 2
        qb = r32 / qa / 2 - qd * qc / qa
    else:
        assert r12 == r21
        qd = np.sqrt(- (r11 + r22) / 2)
        if np.abs(qd) > eps: qb = r31 / qd / 2; qc = r32 / qd / 2
        else: qc = np.sqrt((r22 - r11) / 2); qb = 1 if np.abs(qc) < eps else (r12 / qc / 2)
    return qa, qb, qc, qd
    
def quatern2mat(a, b, c, d):
    return np.array([[a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c], 
                    [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b], 
                    [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b]])
    
def mat2orn(mat): return -mat[0][0], -mat[1][0], mat[2][0], -mat[0][1], -mat[1][1], mat[2][1]
def quatern2orn(*args): return mat2orn(quatern2mat(*args))

def niiheader_to_mat(h):
    R = np.zeros((4, 4))
    R[3][3] = 1
    if h['sform_code'] > 0:
        R[0] = h['srow_x']
        R[1] = h['srow_y']
        R[2] = h['srow_z']
    elif h['qform_code'] == 0:
        R[0][0], R[1][1], R[2][2] = h['pixdim'][1:4]
    else:
        b, c, d = h['quatern_b'], h['quatern_c'], h['quatern_d']
        qx, qy, qz = h['qoffset_x'], h['qoffset_y'], h['qoffset_z']
        qfac, dx, dy, dz = h['pixdim'][:4]
        dz = dz if qfac >= 0 else -dz
        a = np.sqrt(np.clip(1 - (b * b + c * c + d * d), 0, 1))
        R[:3, :3] = quatern2mat(a, b, c, d) * np.array([[dx, dy, dz]])
        R[:3, 3] = qx, qy, qz
    R[:, 0], R[:, 1] = R[:, 1].copy(), R[:, 0].copy()
    return R

def create_nii(dcm, creation):
    data = tp.Tensor(dcm)
    if not isinstance(dcm, DCM) and hasattr(dcm, 'bundle'): bundle = dcm.bundle
    else:
        header = nib.Nifti1Header()
        header['regular'] = b'r'
        header['dim'] = [data.ndim, *data.shape] + [1] * (7 - data.ndim)
        bits = len(data.flatten()[0].tobytes()) * 8
        header['bitpix'] = bits
        header.set_data_dtype(data.numpy().dtype)
        if isinstance(dcm, DCM): meta = dcm.bundle
        else: meta = None
        if meta and 'PixelSpacing' in meta: spacing = [float(x) for x in meta.PixelSpacing]
        else: spacing = [1.0, 1.0]
        if meta and 'SliceThickness' in meta: dz = [float(meta.SliceThickness)]
        else: dz = [1.0]
        header['pixdim'] = [1.0] + spacing + dz + [1.0] * (7 - data.ndim)
        header['qform_code'] = 1
        header['xyzt_units'] = 2
        if meta:
            header['qoffset_x'] = -float(meta.ImagePositionPatient[0])
            header['qoffset_y'] = -float(meta.ImagePositionPatient[1])
            header['qoffset_z'] = float(meta.ImagePositionPatient[2])
            qa, qb, qc, qd = orn2quatern(*[float(x) for x in meta.ImageOrientationPatient])
            header['quatern_b'] = qb
            header['quatern_c'] = qc
            header['quatern_d'] = qd
        bundle = nib.Nifti1Image(data, None, header=header)
    instance = creation(data)
    instance.bundle = bundle
    instance.path = getattr(dcm, 'path', 'Unknown')
    return instance

def nii2dcm(nii, creation):
    data = tp.Tensor(nii)
    if not isinstance(nii, NII) and hasattr(nii, 'bundle'): bundle = nii.bundle
    elif hasattr(nii, 'bundle'):
        header = nii.bundle.header
        if data.ndim > 3: raise TypeError("Dicom is unable to store high dimensional data [%dD]."%data.ndim)
        while data.ndim < 3: data = data.unsqueeze(-1)
        # use_meta_size = header.get('use_meta_size', False)
        # if use_meta_size:
        #     if dimof(data) <= 2: tosize = (self.bundle.Rows, self.bundle.Columns)[:dimof(data)]
        #     else: tosize = (self.bundle.Rows, self.bundle.Columns) + data.shape[2:]
        #     if any([s != 1 for s in scaling]):
        #         raise_rescale()
        #         dt = np.dtype(self.dtype)
        #         mode = 'Nearest' if dt.kind == np.dtype(np.int).kind or dt.kind == np.dtype(np.uint).kind else 'Linear'
        #         data = rescale_to(data.astype(np.float32), tosize, mode = mode).astype(data.dtype)
        # else: tosize = data.shape
        b, c, d = header.get('quatern', (0.0, 0.0, 0.0))
        origin = header.get('origin', (0.0, 0.0, 0.0))
        spacing = header.get('spacing', [1.0] * 8)
        modality = header.get('modality', 'CT')
        if 'header' in header:
            if 'quatern' not in header:
                b, c, d = [header['header'].get('quatern_b', 0.0),
                        header['header'].get('quatern_c', 0.0),
                        header['header'].get('quatern_d', 0.0)]
            if 'origin' not in header:
                origin = [header['header'].get('qoffset_x', 0.0),
                        header['header'].get('qoffset_y', 0.0),
                        header['header'].get('qoffset_z', 0.0)]
            if 'spacing' not in header:
                spacing = header['header'].get('pixdim', [1.0] * 8)
        spacing = spacing[1:4]
        from math import sqrt; a = sqrt(1-b*b-c*c-d*d)
        orn = quatern2orn(a, b, c, d)
        # orn = [-x for x in orn]
        origin = [str(-origin[0]), str(-origin[1]), str(origin[2])]
        slice_thickness = header.get('slice_thickness', spacing[2])
        if 'header' not in header:
            if 'quatern' not in header:
                orn = [float(x) for x in self.bundle.ImageOrientationPatient]
            if 'origin' not in header:
                origin = self.bundle.ImagePositionPatient
            if 'spacing' not in header:
                slice_thickness = float(self.bundle.SliceThickness)
                spacing = [float(x) for x in self.bundle.PixelSpacing] + \
                    [header.get('slice_thickness', abs(slice_thickness))]
            if 'Modality' in self.bundle: modality = self.bundle.Modality
        if 'InstanceCreationTime' in self.bundle: ctime = self.bundle.InstanceCreationTime
        if 'SOPInstanceUID' in self.bundle: UID = self.bundle.SOPInstanceUID
        if 'ContentTime' in self.bundle: time = self.bundle.ContentTime
        if 'TriggerTime' in self.bundle: ttime = self.bundle.TriggerTime
        if 'ReconstructionTargetCenterPatient' in self.bundle:  center = self.bundle.ReconstructionTargetCenterPatient
        bits = len(data.flatten()[0].tobytes()) * 8
        traditional_origin = [-float(origin[0]), -float(origin[1]), float(origin[2])]
        if np.abs(orn[2]) > 0: iz = 0
        elif np.abs(orn[5]) > 0: iz = 1
        else: iz = 2
        position = np.dot(quatern2mat(*orn2quatern(*orn)).T, np.array([traditional_origin]).T)[-1, 0]
        bundles = {}
        typical_slice = float('inf'), None
        Nslice = min(data.shape[-1], header.get('max_slice', float('inf')))
        for slice in range(Nslice):
            # sdcm = dcm.filereader.dcmread(self.bundle.filename, stop_before_pixels=True)
            if not header.get('generate_slices', True) and 0 < slice < Nslice - 1: continue
            sdcm = deepcopy(self.bundle)
            if test(lambda:UID): *segs, tail = UID.split('.')
            if 'SOPInstanceUID' in sdcm:
                sdcm.SOPInstanceUID = '.'.join(segs + [str(int(tail) + slice)])
            if 'ReconstructionTargetCenterPatient' in sdcm and not self.slice_only:
                sdcm.ReconstructionTargetCenterPatient = [0.0, 0.0, center[-1] + slice * slice_thickness]
            if 'TablePosition' in sdcm and not self.slice_only:
                sdcm.TablePosition = position + slice * slice_thickness
            if 'InstanceNumber' in sdcm and not self.slice_only:
                sdcm.InstanceNumber = str(slice + 1)
            if 'ImagePositionPatient' in sdcm and not self.slice_only:
                sdcm.ImagePositionPatient = origin[:iz] + [str(float(origin[iz]) + slice * slice_thickness)] + origin[iz+1:]
            if 'SliceLocation' in sdcm and not self.slice_only:
                sdcm.SliceLocation = str(position + slice * slice_thickness)
            if 'SliceThickness' in sdcm:
                sdcm.SliceThickness = str(abs(slice_thickness))
            if 'InStackPositionNumber' in sdcm and not self.slice_only:
                sdcm.InStackPositionNumber = slice + 1
            if 'ImageOrientationPatient' in sdcm:
                sdcm.ImageOrientationPatient = [str(x) for x in orn]
            # if 'InPlanePhaseEncodingDirection' in sdcm:
            #     del sdcm['InPlanePhaseEncodingDirection']
            if 'Modality' in sdcm and modality:
                sdcm.Modality = modality
            if 'PixelSpacing' in sdcm:
                sdcm.PixelSpacing = [str(x) for x in spacing[:2]]
            if 'BitsStored' in sdcm:
                sdcm.BitsStored = bits
            if 'HighBit' in sdcm:
                sdcm.HighBit = bits - 1
            if 'BitsAllocated' in sdcm:
                sdcm.BitsAllocated = bits
            if 'PixelRepresentation' in sdcm:
                sdcm.PixelRepresentation = int(data.dtype.kind == 'u')
            try:
                self.bundle[0x7005, 0x1018]
                try: sdcm[0x7005, 0x1018]
                except: sdcm[0x7005, 0x1018] = self.bundle[0x7005, 0x1018]
                sdcm[0x7005, 0x1018].value = chr(slice + 1).encode() + chr(0).encode()
            except: pass
            if 'LargestImagePixelValue' in sdcm:
                sdcm.LargestImagePixelValue = np.max(data[..., slice])
            if 'PixelData' in sdcm:
                sdcm.PixelData = data[..., slice].tobytes()
                sdcm['PixelData'].VR = 'OB'
            if 'Rows' in sdcm and 'Columns' in sdcm:
                sdcm.Rows, sdcm.Columns = data.shape[:2]
            if float(sdcm.ImagePositionPatient[2]) < typical_slice[0]:
                typical_slice = float(sdcm.ImagePositionPatient[2]), sdcm
            bundles[slice] = sdcm
        return bundles if header.get('generate_slices', True) else typical_slice[1]

class NII(tp.Tensor):

    def __new__(cls, instance):
        if isinstance(instance, str):
            p = path(instance)
            if not p.ext: p = p // 'nii.gz'
            niiBundle = nib.load(p)
            data = niiBundle.get_data()
            self = super().__new__(cls, data)
            self.bundle = niiBundle
            self.path = p
            return self.transpose_(1, 0)
        elif hasattr(instance, 'shape'):
            if instance.ndim == 0: return instance
            if isinstance(instance, NII): return instance
            return create_nii(instance, lambda x: super().__new__(cls, x))
        else: raise TypeError(f"Unknown input for NII: {instance}. ")

    def __enter__(self): return self

    def __exit__(self, *args): return False

    def __call__(self, data=None):
        if data is None: return self.bundle
        data.bundle = self.bundle
        data.path = self.path
        return NII(data)

    def _create_bundle(self, data, use_header_size=False, spacing=None):
        data = tp.Tensor(data)
        header = self.bundle.header.copy()
        if spacing is not None: header['pixdim'] = [1.0] + list(spacing) + [1.0] * (7 - data.ndim)
        if use_header_size:
            raise NotImplementedError("It appears that the developers forgot to implement keyword use_header_size! Please contact us to remind us. ")
            # if any([s != 1 for s in scaling]):
            #     raise_rescale()
            #     dt = np.dtype(self.dtype)
            #     mode = 'Nearest' if dt.kind == np.dtype(np.int).kind or dt.kind == np.dtype(np.uint).kind else 'Linear'
            #     data = rescale_to(data.astype(np.float32), header['dim'][1: dimof(data) + 1], mode = mode).astype(data.dtype)
        else: header['dim'] = [data.ndim] + list(data.shape) + [1] * (7 - data.ndim)
        return nib.Nifti1Image(data.transpose(1, 0), None, header)

    def save(self, path): nib.save(self.bundle, str(path))

    def save_as_dcm(self, path): DCM(self).save(str(path))

    def update(self):
        self.bundle = self._create_bundle(self, False)

    def save_data(self, data, path, use_header_size=False):
        nib.save(self._create_bundle(data, use_header_size), str(path))

    def spacing(self): return self.bundle.header['pixdim'][1: self.ndim + 1]

    # def resample(self, new_spacing):
    #     raise_rescale()
    #     spacing = self.spacing()
    #     new_spacing = totuple(new_spacing)
    #     if len(new_spacing) == 1: new_spacing *= len(spacing)
    #     dt = np.dtype(self.dtype)
    #     mode = 'Nearest' if dt.kind == np.dtype(np.int).kind or dt.kind == np.dtype(np.uint).kind else 'Linear'
    #     new_spacing = tonumpy(new_spacing)
    #     new_data = rescale_to(self.astype(np.float32), 
    #         tuple(int(x) for x in (np.array(self.shape) * spacing / new_spacing).round()), mode = mode).astype(self.dtype)
    #     instance = super().__new__(NII, new_data.shape, dtype=self.dtype)
    #     instance[...] = new_data
    #     instance.path = self.path
    #     instance.bundle = self._create_bundle(new_data, spacing = new_spacing)
    #     return instance

    def affine(self):
        return niiheader_to_mat(self.bundle.header)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        self = args[0]
        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            collection = []
            ret = tp.Tensor.__torch_function_convert_collect__(ret, collection, cls)
            for r in collection:
                r.bundle = self.bundle
                r.path = self.path
        return ret

class DCM(tp.Tensor):

    def __new__(cls, instance, slice_only=False):
        if isinstance(instance, str):
            p = path(instance)
            if not p.isdir():
                if not slice_only: p = p@path.Folder
            else: slice_only = False
            dcmBundle = dcm.filereader.dcmread(path(__file__)@path.Folder/"template.dcm")
            slice_arrays = {}
            slices = {}
            zs = {}
            readable = False
            direction_down = True
            for p in ([p] if slice_only else p):
                if not p.ext.lower() in ('dcm', 'ima'): continue
                try: image_slice = dcm.filereader.dcmread(p)
                except: continue
                readable = True
                n_slice = int(image_slice.InstanceNumber)
                if 'SeriesNumber' in image_slice: n_series = int(image_slice.SeriesNumber)
                else: n_series = 0
                try: slice_array = image_slice.pixel_array
                except:
                    try:
                        p_dicom = (p@path.Folder//'dicom').mkdir()/p@path.File
                        if not p_dicom.exists():
                            _, stderr = shell(f"dcmdjpeg {p} {p_dicom}")
                        else: stderr = ''
                        if stderr: raise TypeError("Unknown encoding: %s."%p)
                    except: raise TypeError("Unknown encoding: %s."%p)
                    image_slice = dcm.filereader.dcmread(p_dicom)
                    try: slice_array = image_slice.pixel_array
                    except: raise TypeError("Unknown encoding: %s."%p)
                if n_series not in slices:
                    slice_arrays[n_series] = {}
                    slices[n_series] = {}
                    zs[n_series] = {}
                slice_arrays[n_series][n_slice] = slice_array
                slices[n_series][n_slice] = image_slice
                if image_slice.ImageOrientationPatient[2] != 0: iz = 0
                elif image_slice.ImageOrientationPatient[5] != 0: iz = 1
                else: iz = 2
                if 'ImagePositionPatient' in image_slice: z = float(image_slice.ImagePositionPatient[iz])
                elif 'TablePosition' in image_slice: z = image_slice.TablePosition
                elif 'SliceLocation' in image_slice: z = float(image_slice.SliceLocation)
                else: z = 0.
                zs[n_series][n_slice] = z
            if not readable: raise TypeError("Could not create a DICOM object from " + p + ".")
            sorted_series = sorted([(n_series, slices[n_series]) for n_series in slices], key=lambda x: -len(x[1]))
            n_series = sorted_series[0][0]
            possible_series = [s[1] for s in sorted_series if s[0] == n_series]
            if len(possible_series) >= 8: series = possible_series[7]
            elif len(possible_series) >= 3: series = possible_series[2]
            else: series = possible_series[0]
            min_slice = 1000, None
            max_slice = 0, None
            top_slices = -float('inf'), {}
            bottom_slices = float('inf'), {}
            for n_slice in series:
                image_slice = series[n_slice]
                z = zs[n_series][n_slice]
                if n_slice < min_slice[0]:
                    min_slice = n_slice, image_slice
                if n_slice > max_slice[0]:
                    max_slice = n_slice, image_slice
                if z > top_slices[0]:
                    top_slices = z, {n_slice: image_slice}
                if z < bottom_slices[0]:
                    bottom_slices = z, {n_slice: image_slice}
                if z == top_slices[0]:
                    top_slices[1][n_slice] = image_slice
                if z == bottom_slices[0]:
                    bottom_slices[1][n_slice] = image_slice
            N = min(len(top_slices[1].keys()), len(bottom_slices[1].keys()))
            if N >= 8: i_series = 7
            elif N >= 3: i_series = 2
            else: i_series = 0
            bound1 = sorted(top_slices[1].keys())[i_series]
            bound2 = sorted(bottom_slices[1].keys())[i_series]
            if bound1 > bound2:
                zs = {k: v for k, v in zs[n_series].items() if bound2 <= k <= bound1}
                slices = {k: v for k, v in slice_arrays[n_series].items() if bound2 <= k <= bound1}
                max_slice = bound1, top_slices[1][bound1]
                min_slice = bound2, bottom_slices[1][bound2]
            elif bound1 < bound2:
                zs = {k: v for k, v in zs[n_series].items() if bound1 <= k <= bound2}
                slices = {k: v for k, v in slice_arrays[n_series].items() if bound1 <= k <= bound2}
                max_slice = bound2, bottom_slices[1][bound2]
                min_slice = bound1, top_slices[1][bound1]
            else:
                zs = {k: v for k, v in zs[n_series].items()}
                slices = {k: v for k, v in slice_arrays[n_series].items()}
                bound = sorted(series.keys())[0]
                max_slice = min_slice = bound, series[bound]
            direction_down = zs[max_slice[0]] < zs[min_slice[0]]
            typical_slice = max_slice[1] if direction_down else min_slice[1]
            for key in dir(typical_slice):
                if key == 'PixelData' or '_' in key: continue
                if key.capitalize() != key[0] + key[1:].lower(): continue
                dcmBundle[key] = typical_slice[key]
            ozs = tp.Tensor(sorted(zs.values()))
            if len(set(ozs)) > 1:
                volume = tp.stack(orderedValue({zs[i]: slices[i] for i in slices}), -1)
                dcmBundle.SliceThickness = str(tp.abs(tp.mean(ozs[1:] - ozs[:-1])).item())
            else:
                volume = tp.stack(orderedValue(slices), -1)
            volume = volume.astype(toU(volume.dtype) if dcmBundle.PixelRepresentation else toI(volume.dtype))
            dcmBundle.PixelData = volume.tobytes()
            self = super().__new__(cls, volume)
            self.bundle = dcmBundle
            self.path = path
            self.slice_only = slice_only
            self.update()
            return self
        elif hasattr(instance, 'shape'):
            if instance.ndim == 0: return instance
            if isinstance(instance, DCM): return instance
            if isinstance(instance, NII):
                input = nii2dcmBundle(instance)
            else:
                data = tp.Tensor(instance)
                input.path = 'Unknown'
                self.slice_only = False
            self.update()
            return self
        else: raise TypeError(f"Unknown input for DCM: {instance}. ")

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __call__(self, data=None):
        if data is None: return self.bundle
        data.bundle = self.bundle
        data.path = self.path
        return DCM(data)
    def _create_bundles(self, data, **header):
        '''
        header: a dict containing 'quatern', 'origin', 'spacing',
            'slice_thickness', 'use_meta_size', 'max_slice',
            'generate_slices', 'modality', 'header'
        '''
        if 'header' not in header and isinstance(data, NII):
            header['header'] = data().header
        data = np.array(data)
        # data = data.transpose(1, 0, *range(2, dimof(data)))
        if dimof(data) > 3: raise TypeError("Dicom is unable to store high dimensional data [%dD]."%dimof(data))
        while dimof(data) < 3: data = np.expand_dims(data, -1)
        use_meta_size = header.get('use_meta_size', False)
        if use_meta_size:
            if dimof(data) <= 2: tosize = (self.bundle.Rows, self.bundle.Columns)[:dimof(data)]
            else: tosize = (self.bundle.Rows, self.bundle.Columns) + data.shape[2:]
            if any([s != 1 for s in scaling]):
                raise_rescale()
                dt = np.dtype(self.dtype)
                mode = 'Nearest' if dt.kind == np.dtype(np.int).kind or dt.kind == np.dtype(np.uint).kind else 'Linear'
                data = rescale_to(data.astype(np.float32), tosize, mode = mode).astype(data.dtype)
        else: tosize = data.shape
        b, c, d = header.get('quatern', (0.0, 0.0, 0.0))
        origin = header.get('origin', (0.0, 0.0, 0.0))
        spacing = header.get('spacing', [1.0] * 8)
        modality = header.get('modality', 'CT')
        if 'header' in header:
            if 'quatern' not in header:
                b, c, d = [header['header'].get('quatern_b', 0.0),
                        header['header'].get('quatern_c', 0.0),
                        header['header'].get('quatern_d', 0.0)]
            if 'origin' not in header:
                origin = [header['header'].get('qoffset_x', 0.0),
                        header['header'].get('qoffset_y', 0.0),
                        header['header'].get('qoffset_z', 0.0)]
            if 'spacing' not in header:
                spacing = header['header'].get('pixdim', [1.0] * 8)
        spacing = spacing[1:4]
        from math import sqrt; a = sqrt(1-b*b-c*c-d*d)
        orn = quatern2orn(a, b, c, d)
        # orn = [-x for x in orn]
        origin = [str(-origin[0]), str(-origin[1]), str(origin[2])]
        slice_thickness = header.get('slice_thickness', spacing[2])
        if 'header' not in header:
            if 'quatern' not in header:
                orn = [float(x) for x in self.bundle.ImageOrientationPatient]
            if 'origin' not in header:
                origin = self.bundle.ImagePositionPatient
            if 'spacing' not in header:
                slice_thickness = float(self.bundle.SliceThickness)
                spacing = [float(x) for x in self.bundle.PixelSpacing] + \
                    [header.get('slice_thickness', abs(slice_thickness))]
            if 'Modality' in self.bundle: modality = self.bundle.Modality
        if 'InstanceCreationTime' in self.bundle: ctime = self.bundle.InstanceCreationTime
        if 'SOPInstanceUID' in self.bundle: UID = self.bundle.SOPInstanceUID
        if 'ContentTime' in self.bundle: time = self.bundle.ContentTime
        if 'TriggerTime' in self.bundle: ttime = self.bundle.TriggerTime
        if 'ReconstructionTargetCenterPatient' in self.bundle:  center = self.bundle.ReconstructionTargetCenterPatient
        bits = len(data.flatten()[0].tobytes()) * 8
        traditional_origin = [-float(origin[0]), -float(origin[1]), float(origin[2])]
        if np.abs(orn[2]) > 0: iz = 0
        elif np.abs(orn[5]) > 0: iz = 1
        else: iz = 2
        position = np.dot(quatern2mat(*orn2quatern(*orn)).T, np.array([traditional_origin]).T)[-1, 0]
        bundles = {}
        typical_slice = float('inf'), None
        Nslice = min(data.shape[-1], header.get('max_slice', float('inf')))
        for slice in range(Nslice):
            # sdcm = dcm.filereader.dcmread(self.bundle.filename, stop_before_pixels=True)
            if not header.get('generate_slices', True) and 0 < slice < Nslice - 1: continue
            sdcm = deepcopy(self.bundle)
            if test(lambda:UID): *segs, tail = UID.split('.')
            if 'SOPInstanceUID' in sdcm:
                sdcm.SOPInstanceUID = '.'.join(segs + [str(int(tail) + slice)])
            if 'ReconstructionTargetCenterPatient' in sdcm and not self.slice_only:
                sdcm.ReconstructionTargetCenterPatient = [0.0, 0.0, center[-1] + slice * slice_thickness]
            if 'TablePosition' in sdcm and not self.slice_only:
                sdcm.TablePosition = position + slice * slice_thickness
            if 'InstanceNumber' in sdcm and not self.slice_only:
                sdcm.InstanceNumber = str(slice + 1)
            if 'ImagePositionPatient' in sdcm and not self.slice_only:
                sdcm.ImagePositionPatient = origin[:iz] + [str(float(origin[iz]) + slice * slice_thickness)] + origin[iz+1:]
            if 'SliceLocation' in sdcm and not self.slice_only:
                sdcm.SliceLocation = str(position + slice * slice_thickness)
            if 'SliceThickness' in sdcm:
                sdcm.SliceThickness = str(abs(slice_thickness))
            if 'InStackPositionNumber' in sdcm and not self.slice_only:
                sdcm.InStackPositionNumber = slice + 1
            if 'ImageOrientationPatient' in sdcm:
                sdcm.ImageOrientationPatient = [str(x) for x in orn]
            # if 'InPlanePhaseEncodingDirection' in sdcm:
            #     del sdcm['InPlanePhaseEncodingDirection']
            if 'Modality' in sdcm and modality:
                sdcm.Modality = modality
            if 'PixelSpacing' in sdcm:
                sdcm.PixelSpacing = [str(x) for x in spacing[:2]]
            if 'BitsStored' in sdcm:
                sdcm.BitsStored = bits
            if 'HighBit' in sdcm:
                sdcm.HighBit = bits - 1
            if 'BitsAllocated' in sdcm:
                sdcm.BitsAllocated = bits
            if 'PixelRepresentation' in sdcm:
                sdcm.PixelRepresentation = int(data.dtype.kind == 'u')
            try:
                self.bundle[0x7005, 0x1018]
                try: sdcm[0x7005, 0x1018]
                except: sdcm[0x7005, 0x1018] = self.bundle[0x7005, 0x1018]
                sdcm[0x7005, 0x1018].value = chr(slice + 1).encode() + chr(0).encode()
            except: pass
            if 'LargestImagePixelValue' in sdcm:
                sdcm.LargestImagePixelValue = np.max(data[..., slice])
            if 'PixelData' in sdcm:
                sdcm.PixelData = data[..., slice].tobytes()
                sdcm['PixelData'].VR = 'OB'
            if 'Rows' in sdcm and 'Columns' in sdcm:
                sdcm.Rows, sdcm.Columns = data.shape[:2]
            if float(sdcm.ImagePositionPatient[2]) < typical_slice[0]:
                typical_slice = float(sdcm.ImagePositionPatient[2]), sdcm
            bundles[slice] = sdcm
        return bundles if header.get('generate_slices', True) else typical_slice[1]
    def save(self, path): self.save_data(self, str(path))
    def save_as_nii(self, path): NII(self).save(str(path))
    def update(self, header=None):
        if not isinstance(self.bundle, dcm.dataset.Dataset):
            self.bundle = dcm.filereader.dcmread(Template)
        if header: self.bundle = self._create_bundles(self, generate_slices=False, header=header)
        else: self.bundle = self._create_bundles(self, generate_slices=False)
    def save_data(self, data, fpath, **header):
        fpath = str(fpath)
        if os.path.isfile(fpath):
            path = os.path.dirname(fpath)
            pmkdir(path)
        else:
            pmkdir(fpath)
            if '.' in os.path.basename(fpath):
                path = os.path.dirname(fpath)
            else:
                path = fpath
                fpath = os.path.join(path, "slice.dcm")
        slices = self._create_bundles(data, **header)
        if self.slice_only: slices[0].save_as(fpath); return
        for i, s in slices.items():
            *pfix, sfix = fpath.split('.')
            s.save_as('.'.join(pfix) + "_%04d"%(i + 1) + '.' + sfix)
    def spacing(self):
        meta = self()
        if 'PixelSpacing' in meta: spacing = [float(x) for x in meta.PixelSpacing]
        else: spacing = [1.0, 1.0]
        if 'SliceThickness' in meta: dz = [float(meta.SliceThickness)]
        else: dz = [1.0]
        spacing = spacing + dz
        return np.array(spacing)
    def resample(self, new_spacing):
        raise_rescale()
        spacing = self.spacing()
        new_spacing = totuple(new_spacing)
        if len(new_spacing) == 1: new_spacing *= len(spacing)
        dt = np.dtype(self.dtype)
        mode = 'Nearest' if dt.kind == np.dtype(np.int).kind or dt.kind == np.dtype(np.uint).kind else 'Linear'
        new_spacing = tonumpy(new_spacing)
        new_data = rescale_to(self.astype(np.float32), 
            tuple(int(x) for x in (np.array(self.shape) * spacing / new_spacing).round()), mode = mode).astype(self.dtype)
        instance = super().__new__(DCM, new_data.shape, dtype=self.dtype)
        instance[...] = new_data
        instance.path = self.path
        instance.slice_only = self.slice_only
        instance.bundle = self._create_bundle(new_data, spacing = new_spacing)
        return instance
    def affine(self):
        return niiheader_to_mat(NII(self)().header)
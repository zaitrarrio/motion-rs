"""Read a torch.save pickle (zip or legacy) without torch; print arrays as JSON."""
import sys, json, zipfile, pickle, io, struct
import numpy as np

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_, "HalfStorage": np.float16}

def load(path):
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        names = zf.namelist()
        prefix = names[0].split('/')[0]
        storages = {}
        class U(pickle.Unpickler):
            def find_class(self, mod, name):
                if name == "_rebuild_tensor_v2":
                    def rebuild(storage, offset, size, stride, *a):
                        arr, dt = storage
                        n = int(np.prod(size)) if size else 1
                        flat = np.frombuffer(arr, dtype=dt)
                        # assume contiguous
                        return flat[offset:offset+n].reshape(size).copy()
                    return rebuild
                if name in DTYPES:
                    return name
                if mod == "collections" and name == "OrderedDict":
                    import collections; return collections.OrderedDict
                if mod == "torch" and name == "Size": return tuple
                return super().find_class(mod, name)
            def persistent_load(self, pid):
                typ, cls, key, loc, numel = pid
                data = zf.read(f"{prefix}/data/{key}")
                return (data, DTYPES[cls])
        return U(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    raise SystemExit("legacy format not handled")

obj = load(sys.argv[1])
def conv(o):
    if isinstance(o, np.ndarray): return {"shape": list(o.shape), "data": o.astype(float).ravel().tolist()}
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    return o
print(json.dumps(conv(obj)))

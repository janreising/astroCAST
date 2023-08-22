import pytest
import dask

from astroCAST.preparation import *

class Test_Delta:

    @pytest.mark.parametrize("input_type", [np.ndarray, "testdata/sample_0.tiff", "testdata/sample_0.h5", "tiledb"])
    @pytest.mark.parametrize("in_memory", (True, False))
    @pytest.mark.parametrize("parallel", (True, False))
    def test_load(self, input_type, in_memory, parallel):

        Z, X, Y = 50, 10, 10

        if input_type == np.ndarray:
            data = np.random.randint(0, 100, (Z, X, Y), dtype=int)
            loc = None

        elif input_type == "tiledb":

            arr = da.from_array(
                x=np.random.randint(0, 100, (Z, X, Y), dtype=int),
                chunks=(Z, "auto", "auto")
            )

            loc = None
            tmpdir = Path(tempfile.mkdtemp()).joinpath("temp.tdb")
            logging.warning(f"tmpdir: {tmpdir}, {type(tmpdir)}")

            arr.to_tiledb(tmpdir.as_posix())
            data = tmpdir

        elif isinstance(input_type, str):

            path = Path(input_type)
            assert path.is_file()
            data = path

            loc = "data/ch0" if path.suffix == ".h5" else None

        else:
            raise TypeError

        delta = Delta(data, loc=loc, in_memory=in_memory, parallel=parallel)

        delta.run(method="background", window=5)

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("parallel", (True, False))
    @pytest.mark.parametrize("use_dask", (True, False))
    @pytest.mark.parametrize("in_memory", (True, False))
    def test_methods_run(self, method, parallel, use_dask, in_memory):

        Z, X, Y = 25, 2, 2

        data = np.random.randint(0, 100, (Z, X, Y), dtype=int)
        loc = None

        delta = Delta(data, loc=loc,
                      in_memory=in_memory, parallel=parallel)

        delta.run(method=method, window=5, use_dask=use_dask)

    @pytest.mark.parametrize("dim", [(100), (100, 5), (100, 5, 5), (100, 2, 10)])
    def test_background_dimensions(self, dim):

        arr = np.random.randint(0, 100, dim, dtype=int)
        orig_shape = arr.shape

        res = Delta.calculate_delta_min_filter(arr, window=10)

        assert res.shape == orig_shape, f"dimensions are not the same input: {dim} vs output: {res.shape}"

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("parallel", (True, False))
    @pytest.mark.parametrize("use_dask", (True, False))
    @pytest.mark.parametrize("in_memory", (True, False))
    def test_result_for_parallel(self, method, parallel, use_dask, in_memory):

        dim = (250, 50, 50)
        window = 10

        arr = np.random.randint(0, 100, dim, dtype=int)

        ctrl = Delta.calculate_delta_min_filter(arr.copy(), window, method=method)
        logging.warning(f"sum of ctrl: {np.sum(ctrl)}")

        delta = Delta(arr, loc=None, in_memory=in_memory, parallel=parallel)
        res = delta.run(method=method, window=window, use_dask=use_dask, overwrite_first_frame=False)

        assert np.allclose(ctrl, res)

    @pytest.mark.xfail(reason="Not implemented")
    def test_quality_of_dff(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Not implemented")
    def test_new_dFF_version(self):
        raise NotImplementedError

class Test_Input:

    @pytest.mark.parametrize("num_files", [1, 12])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_convert_single_tiff_series(self, num_files, in_memory):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = []
            for n in range(num_files):
                img = np.random.random((1, 10, 10))
                images.append(img)

                tifffile.imwrite(tmpdir.joinpath(f"ss_single_{n}.tiff"), img)

            img_stack = np.stack(images)
            img_stack = np.squeeze(img_stack)

            # Loaded
            inp = Input()

            tmpdir = list(tmpdir.glob("*"))[0] if num_files == 1 else tmpdir
            stack = inp.run(input_path=tmpdir, dtype=None, in_memory=in_memory)
            stack = stack["ch0"]

            img_stack = np.squeeze(img_stack)
            stack = np.squeeze(stack)

            assert img_stack.shape == stack.shape
            assert np.array_equal(img_stack, stack)

    @pytest.mark.parametrize("num_channels", [2, 3])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_convert_multi_channel(self, num_channels, in_memory):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(7):
                for n in range(num_channels):

                    img = np.random.random((1, 10, 10))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            # Loaded
            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels, dtype=None, in_memory=in_memory)

            for ch in images.keys():

                ref = np.squeeze(images[ch])
                res = np.squeeze(stack[ch])

                assert ref.shape == res.shape
                assert np.array_equal(ref, res)

    @pytest.mark.parametrize("num_channels", [1, 2, 3])
    @pytest.mark.parametrize("subtract_background", ["arr"])
    @pytest.mark.parametrize("subtract_func", [np.mean, "mean", "std", "min", "max"])
    def test_subtract(self, num_channels, subtract_background, subtract_func, in_memory=True):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            X, Y = 10, 10

            # Reference
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(7):
                for n in range(num_channels):

                    img = np.random.random((1, X, Y))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            if num_channels == 1:
                subtract_background = np.random.random((X, Y))

            else:
                subtract_background = "ch0"

            # Loaded
            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels,
                                  subtract_background=subtract_background, subtract_func=subtract_func,
                                  dtype=None, in_memory=in_memory)

            # check result
            func_reduction = {"mean": np.mean, "std": np.std, "min": np.min, "max":np.max}

            if num_channels == 1:

                assert np.array_equal(
                    stack["ch0"],
                    images["ch0"] - subtract_background
                )

            else:

                func = func_reduction[subtract_func] if not callable(subtract_func) else subtract_func
                background = func(images["ch0"], axis=0)

                for ch in stack.keys():

                    if ch == "ch0":
                        pass

                    ctrl = images[ch] - background
                    res = stack[ch]

                    assert ctrl.shape == res.shape, f"dimensions are not equal: {ctrl.shape} vs. {res.shape}"
                    assert np.allclose(res, ctrl), "values are not equal"

    @pytest.mark.parametrize("rescale", [1, 0.5, 20, (0.5, 0.3), (20, 15)])
    def test_resize(self, rescale, num_channels=1):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            Z, X, Y = 20, 12, 12

            # Reference
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(20):
                for n in range(num_channels):

                    img = np.random.random((1, X, Y))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels, rescale=rescale,
                                  subtract_background=None, subtract_func=None,
                                  dtype=None, in_memory=True)

            res = stack["ch0"]

            if rescale == 1:
                assert res.shape == (Z, X, Y)

            elif isinstance(rescale, float):
                assert res.shape == (Z, int(X*rescale), int(Y*rescale))

            elif isinstance(rescale, int):
                assert res.shape == (Z, rescale, rescale)

            elif isinstance(rescale, tuple):
                rx, ry = rescale

                if isinstance(rx, int):
                    assert res.shape == (Z, rx, ry)

                elif isinstance(rx, float):

                    assert res.shape == (Z, int(X*rx), int(Y*ry))

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff"])
    @pytest.mark.parametrize("chunks", [None, (5, 5, 5)])
    def test_output(self, output_path, chunks):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            data = {0: np.random.random((25, 10, 10))}

            inp = Input()
            inp.save(output_path, data, chunks=chunks)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff"])
    def test_intput_output(self, output_path):

        num_files = 25

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = []
            for n in range(num_files):
                img = np.random.random((1, 10, 10))
                images.append(img)

                tifffile.imwrite(tmpdir.joinpath(f"ss_single_{n}.tiff"), img)

            img_stack = np.stack(images)
            img_stack = np.squeeze(img_stack)

            # Loaded
            inp = Input()

            tmpdir = list(tmpdir.glob("*"))[0] if num_files == 1 else tmpdir
            output_path = tmpdir.joinpath(output_path)

            inp.run(input_path=tmpdir, output_path=output_path,
                            dtype=None, in_memory=False, h5_loc="data")

            assert output_path.is_file() or output_path.is_dir(), f"cannot find output file: {output_path}"

            # load back
            if output_path.suffix == ".h5":

                with h5py.File(output_path.as_posix(), "r") as f:
                    res = f["data/ch0"][:]

            elif output_path.suffix == ".tiff":
                res = tifffile.imread(output_path.as_posix())

            elif output_path.suffix == ".tdb":
                res = tiledb.open(output_path.as_posix())

            else:
                raise NotImplementedError

            res = np.squeeze(res)

            assert img_stack.shape == res.shape
            assert np.array_equal(img_stack, res)

class Test_IO:

    @pytest.mark.parametrize("prefix", ["", "00000"])
    @pytest.mark.parametrize("sep", ["_", "x", "-"])
    def test_alphanumerical_names(self, prefix, sep):

        names = []
        for n in range(1000):
            name = f"img{sep}{prefix}{n}.ext"
            names.append(name)

        names_shuffled = np.random.shuffle(names.copy())

        assert names != names_shuffled, "randomization did not work"

        io = IO()
        names_sorted = io.sort_alpha_numerical_names(names, sep=sep)

        assert names == names_sorted, "sorting did not work"

    @pytest.mark.xfail
    def test_load_czi(self, output_path="out.czi", shape=(10, 5, 5)):
        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix)

            arr_load = io.load(output_path, h5_loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    def test_save_load(self, output_path, shape):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix)

            arr_load = io.load(output_path, h5_loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)
    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    @pytest.mark.parametrize("chunks", [None, (2, 2, 2), "infer"])
    def test_save_chunks(self, shape, chunks, output_path="out.h5"):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix, chunks=chunks)

            arr_load = io.load(output_path, h5_loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("compression", ["gzip", "infer"])
    @pytest.mark.parametrize("shape", [(100, 100, 100), (512, 512, 5000)])
    def test_save_compression(self, compression, shape, chunks=(100, 100, 100), output_path="out.h5"):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.zeros(shape, dtype=float)

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix, chunks=chunks, compression=compression)

            arr_load = io.load(output_path, h5_loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    def test_z_slice(self, output_path, shape, z_slice=(2, 8)):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            z0, z1 = z_slice
            original_array = arr[z0:z1, :, :]

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix)

            arr_load = io.load(output_path, h5_loc=h5loc, z_slice=z_slice)

            assert original_array.shape == arr_load.shape
            assert np.array_equal(original_array, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    def test_lazy_load(self, output_path, shape=(100, 100, 100)):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Saving
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0":arr}

            output_path = io.save(output_path, data, h5_loc=prefix)
            logging.warning(output_path)

            # Loading
            arr_load = io.load(output_path, h5_loc=h5loc, lazy=True)

            assert isinstance(arr_load, (dask.array.Array, dask.array.core.Array)), f"type: {type(arr_load)}"
            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("sep", ["_", "x"])
    def test_load_sequential_tiff(self, sep, shape=(100, 100, 100)):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            input_dir = tmpdir.joinpath("seq_tiff")
            input_dir.mkdir()

            # create tiff files
            arr = np.random.random(shape)

            for z in range(len(arr)):
                img = arr[z, :, :]
                tifffile.imwrite(input_dir.joinpath(f"img{sep}{z}.tiff"), data=img)

            # load data
            io = IO()
            arr_load = io.load(input_dir, sep=sep)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

class Test_MotionCorrection:

    @pytest.mark.parametrize("input_type", ["array", ".h5", ".tdb", ".tiff"])
    def test_random(self, input_type, shape=(100, 100, 100)):

        data = np.random.random(shape)
        h5_loc = None

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            io = IO()

            if input_type == ".h5":

                h5_loc = "mc/ch0"
                temp_path = tmpdir.joinpath("test.h5")
                io.save(temp_path, data={"ch0":data}, h5_loc="mc")

                data = temp_path

            elif input_type == ".tiff":

                temp_path = tmpdir.joinpath("test.tiff")
                io.save(temp_path, data={"ch0":data})

                data = temp_path

            elif input_type == ".tdb":

                temp_path = tmpdir.joinpath("test.tdb")
                temp_path = io.save(temp_path, data={"test/ch0":data}, h5_loc=None)

                assert temp_path.is_dir(), f"cannot find {temp_path}"
                data = temp_path

            elif input_type == "array":
                pass

            else:
                raise ValueError

            mc = MotionCorrection()
            mc.run(data, h5_loc=h5_loc, max_shifts=(6, 6))

            data = mc.save(output=None, remove_mmap=True)
            assert type(data) == np.ndarray


    @pytest.mark.parametrize("input_type", [".tdb"])
    @pytest.mark.skip("currently doesn't work. revisit later.")
    def test_random_tdb(self, input_type, shape=(100, 100, 100)):

        data = np.random.random(shape)
        h5_loc = None

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            io = IO()

            if input_type == ".h5":

                h5_loc = "mc/ch0"
                temp_path = tmpdir.joinpath("test.h5")
                io.save(temp_path, data={"test/ch0":data}, h5_loc="mc")

                data = temp_path

            elif input_type == ".tiff":

                temp_path = tmpdir.joinpath("test.tiff")
                io.save(temp_path, data={"ch0":data})

                data = temp_path

            elif input_type == ".tdb":

                temp_path = tmpdir.joinpath("test.tdb")
                temp_path = io.save(temp_path, data={"ch0":data}, h5_loc=None)

                assert temp_path.is_file(), f"cannot find {temp_path}"
                data = temp_path

            elif input_type == "array":
                pass

            else:
                raise ValueError

            mc = MotionCorrection()
            mc.run(data, h5_loc=h5_loc, max_shifts=(6, 6))

            data = mc.save(output=None, remove_mmap=True)
            assert type(data) == np.ndarray

    @pytest.mark.parametrize("param", [{"input_":"testdata/sample_0.tiff"},
                                            {"input_":"testdata/sample_0.h5", "h5_loc":"data/ch0"}])
    def test_real_input(self, param):
        mc = MotionCorrection()
        mc.run(**param, max_shifts=(6, 6))

        data = mc.save(output=None, remove_mmap=True)
        assert type(data) == np.ndarray

    @pytest.mark.parametrize("video_param",
        [{"speed": 0, "size": (100, 50, 50)}, {"speed": 0.1, "size": (100, 50, 50)},
         {"speed": 0.01, "size": (1000, 250, 250)}, {"speed": 0.01, "size": (1000, 250, 250)}])
    def test_motion_correct_performance(self, video_param, dtype=np.uint8):

        motion_speed = video_param["speed"]
        Z, X, Y = video_param["size"]

        # Generate random structure
        data = np.zeros((Z, X, Y), dtype=dtype)
        structure = np.random.randint(low=0, high=255, size=(X, Y), dtype=dtype)

        # Add motion to each frame
        for t in range(Z):
            shift = int(t * motion_speed)
            shifted_structure = np.roll(structure, shift, axis=(0, 1))
            data[t] = shifted_structure

        mc = MotionCorrection()
        mc.run(data, h5_loc=None, max_shifts=(int(X/2)-1, int(Y/2)-1))

        data = mc.save(output=None, remove_mmap=True)
        assert type(data) == np.ndarray

        # get average shift per frame
        mcs = np.array(mc.shifts)[:, 0]
        mcs = np.mean(np.abs(np.diff(mcs)))

        assert np.allclose(mcs, motion_speed, rtol=5)

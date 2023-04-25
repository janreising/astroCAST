def pytest_generate_tests(metafunc):

    if "file_path" in metafunc.fixturenames:
        metafunc.parametrize("file_path", ["testdata/sample_0.tiff", "testdata/sample_0.h5"])

    if "pre_post_frame" in metafunc.fixturenames:
        metafunc.parametrize("pre_post_frame", [5, (3, 2)])

    if "normalize" in metafunc.fixturenames:
        metafunc.parametrize("normalize", [None, "local", "global"])

    if "typ" in metafunc.fixturenames:
        metafunc.parametrize("typ", ["dataframe", "list", "array"])

    if "ragged" in metafunc.fixturenames:
        metafunc.parametrize("ragged", [True, False])

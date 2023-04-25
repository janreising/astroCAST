def pytest_generate_tests(metafunc):

    if "file_path" in metafunc.fixturenames:
        metafunc.parametrize("file_path", ["testdata/sample_0.tiff", "testdata/sample_0.h5"])

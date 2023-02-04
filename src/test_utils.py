import pathlib
import subprocess
# import sys

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent / "test_files"

def download_test_data():
    subprocess.run(
        [
            "girder-client",
            "--api-url",
            "https://data.kitware.com/api/v1",
            "localsync",
            "62a0efe5bddec9d0c4175c1f",
            str(TEST_DATA_DIR),
        ],
        #stdout=sys.stdout,
    )

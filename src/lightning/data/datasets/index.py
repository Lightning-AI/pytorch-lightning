import math
import os
from io import TextIOWrapper

from lightning.app.utilities.network import LightningClient


def get_index(s3_connection_path: str, index_file_path: str) -> bool:
    """Creates an index of file paths that are in the provided s3 path.

    Returns:
        Returns True is the index got created and False if it wasn't
    """

    if s3_connection_path.startswith("/data/"):
        s3_connection_path = s3_connection_path[len("/data/") :]
    if s3_connection_path.startswith("s3://"):
        s3_connection_path = s3_connection_path[len("s3://") :]

    try:
        index_exists = _get_index(s3_connection_path, index_file_path)
    except KeyError:
        index_exists = False
    except Exception as exc:
        raise ValueError(f"Could not get index file with error: {exc}")

    # Fallback to creating an index from scratch
    if not index_exists:
        index_exists = _create_index(s3_connection_path, index_file_path)

    return index_exists


def _create_index_recursive(root: str, write_to: TextIOWrapper) -> None:
    """Recursively pull files from s3 prefixes until full path is available."""
    from fsspec.core import url_to_fs
    from torchdata.datapipes.iter import FSSpecFileLister

    files = FSSpecFileLister(root).list_files_by_fsspec()

    for file in files:
        if file == root:
            continue

        fs, path = url_to_fs(file)

        if not fs.isfile(file):
            _create_index_recursive(root=file, write_to=write_to)
        else:
            write_to.write(file + "\n")


def _create_index(data_connection_path: str, index_file_path: str) -> bool:
    """Fallback mechanism for index creation."""
    from botocore.exceptions import NoCredentialsError

    print(f"Creating Index for {data_connection_path} in {index_file_path}")
    try:
        list_from = f"s3://{data_connection_path}" if not os.path.isdir(data_connection_path) else data_connection_path

        if not os.path.exists(os.path.dirname(index_file_path)):
            os.makedirs(os.path.dirname(index_file_path))

        with open(index_file_path, "w") as f:
            _create_index_recursive(root=list_from, write_to=f)

        return True
    except NoCredentialsError as exc:
        print(
            "Unable to locate credentials. \
            Make sure you have set the following environment variables: \nAWS_ACCESS_KEY\\AWS_SECRET_ACCESS_KEY"
        )
        os.remove(index_file_path)
        raise ValueError(exc)
    except Exception as exc:
        os.remove(index_file_path)
        raise ValueError(exc)


def _get_index(data_connection_path: str, index_file_path: str) -> bool:
    """Expecting a string in the format s3:// or /data/...

    Returns:
        True if the index retrieved
    """

    PROJECT_ID_ENV = "LCP_ID"

    client = LightningClient(retry=False)

    if PROJECT_ID_ENV in os.environ:
        project_id = os.environ[PROJECT_ID_ENV]
    else:
        return False

    try:
        cluster_bindings = client.projects_service_list_project_cluster_bindings(project_id).clusters

        # For now just use the first one
        # For BYOC we will have to update this
        cluster = cluster_bindings[0]

        # Find the data connection object first
        data_connections = client.data_connection_service_list_data_connections(project_id).data_connections
        data_connection = [con for con in data_connections if con.name == data_connection_path]

        if len(data_connection) == 1:
            print(f"Placing existing index for {data_connection_path} in {index_file_path}")

            data_connection = data_connection[0]
            # Then use the ID of the data connection for retrieving the index
            folder_index = client.data_connection_service_get_data_connection_folder_index(
                project_id=project_id, id=data_connection.id, cluster_id=cluster.cluster_id
            )

            # Compute number of pages we need to retrieve
            num_pages = math.ceil(int(folder_index.nested_file_count) / folder_index.page_size)

            # Get all the pages and append to the index
            with open(index_file_path, "a") as f:
                f.truncate(0)

                for page_num in range(num_pages):
                    page = client.data_connection_service_get_data_connection_artifacts_page(
                        project_id=project_id,
                        id=data_connection.id,
                        cluster_id="litng-ai-03",
                        page_number=str(page_num),
                    ).artifacts

                    f.writelines([f"s3://{data_connection_path}/{item.filename}" + "\n" for item in page])
            return True
        return False

    except Exception:
        return False

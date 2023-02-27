import os
import subprocess
import jupytext
import filecmp
from typing import List


def get_changed_files_ending(ending: str, git_filter: str = "ACMR") -> List[str]:
    """Returns a list of paths to all changed files with the given ending

    Args:
        ending (str): end of file string to match
        git_filter (str, optional): filter to pass to git diff command. Defaults to "ACMR".

    Returns:
        List[str]: list of paths to changed files
    """

    cmd = f"git diff --name-only --cached --diff-filter={git_filter}"
    output = str(subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0])
    changed_paths = str(output)[2:].split("\\n")
    file_paths = [path for path in changed_paths if path.endswith(ending)]

    return file_paths


def get_renamed_files_ending(ending: str) -> List[str]:
    """Return a list of paths to all renamed files with the given ending

    Args:
        ending (str): end of file string to match

    Returns:
        List[str]: list of paths to renamed files
    """

    cmd = f"git status -s"
    output = str(subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0])
    change_lines = str(output)[2:].split("\\n")
    change_lines = [line for line in change_lines if line.endswith(ending)]
    renamed_lines = [line[3:] for line in change_lines if line.startswith("R ")]
    renamed_files = [
        {"Old": line.split(" -> ")[0], "New": line.split(" -> ")[1]}
        for line in renamed_lines
    ]

    return renamed_files


def get_py_path_from_notebook_path(nb_path: str) -> str:
    """Returns path to python file given the associated notebook file path

    Args:
        nb_path (str): path of the notebook file

    Returns:
        str: path to the python file
    """

    nb_split = os.path.split(nb_path)
    py_path = os.path.join(
        nb_split[0], "py_versions", nb_split[1].replace(".ipynb", ".py")
    )
    py_version_dir = os.path.split(py_path)
    if not os.path.exists(py_version_dir[0]):
        os.makedirs(py_version_dir[0])

    return py_path


def delete_py_versions_of_deleted_notebooks():
    """Find deleted notebook files and remove their associated python files"""

    nb_file_paths = get_changed_files_ending(".ipynb", git_filter="D")

    for nb_path in nb_file_paths:
        py_path = get_py_path_from_notebook_path(nb_path)

        cmd = f"git rm {py_path}"
        subprocess.Popen(cmd).communicate()

        print(f"{py_path}: Deleted to match {nb_path}")


def rename_py_versions_of_renamed_notebooks():
    """Find renamed notebook files and rename their associated python files"""

    nb_file_paths = get_renamed_files_ending(".ipynb")

    for nb_pair in nb_file_paths:
        old_nb_path = nb_pair["Old"]
        new_nb_path = nb_pair["New"]
        old_py_path = get_py_path_from_notebook_path(old_nb_path)
        new_py_path = get_py_path_from_notebook_path(new_nb_path)

        cmd = f"git mv {old_py_path} {new_py_path}"
        subprocess.Popen(cmd).communicate()

        print(f"{old_py_path} -> {new_py_path}: Renamed to match {new_nb_path}")


def make_py_versions_of_notebooks():
    """Make python version of notebook files changed in git for this commit"""

    nb_file_paths = get_changed_files_ending(".ipynb", git_filter="ACMR")

    for nb_path in nb_file_paths:
        py_path = get_py_path_from_notebook_path(nb_path)

        notebook = jupytext.read(nb_path)
        jupytext.write(notebook, py_path, fmt="py:percent")

        cmd = f"git add {py_path}"
        subprocess.Popen(cmd).communicate()

        print(f"{py_path}: Updated from {nb_path}")


def format_files_ending(ending_list: List[str]):
    """Format files with the given endings

    Args:
        ending_list (List[str]): endings of files to format
    """
    # Loop through the given endings
    for ending in ending_list:
        # Get files with the given ending
        # ACM is added, changed or moved - We don't need R - removed as the file will no longer be tracked
        file_list = get_changed_files_ending(ending, git_filter="ACM")

        for file in file_list:
            # Format the file with black
            cmd = f"python -m black {file}"
            subprocess.Popen(cmd).communicate()

            # Re-add the file to git now that it is formatted
            cmd = f"git add {file}"
            subprocess.Popen(cmd).communicate()


def export_conda_env_to_file(file_name: str = "environment.yml"):
    """Export the conda environment to file with the given file name

    Args:
        file_name (str, optional): name for the export file. Defaults to "environment.yml".
    """

    env_name = get_conda_env_name()

    # export the conda environment
    cmd = f"conda env export -n {env_name} -f {file_name}"
    subprocess.Popen(cmd).communicate()

    # Remove the last line of the file
    # This line gives the path to the environment which will be different for each user.
    # We don't want this line changing just because someone different committed something.
    # So we remove this line
    with open(file_name, "r") as file:
        lines = file.readlines()

    lines = lines[:-1]
    with open(file_name, "w") as file:
        file.writelines(lines)


def update_conda_env_file(file_name: str = "environment.yml"):
    """update the conda environment file

    Args:
        file_name (str, optional): name for the export file. Defaults to "environment.yml".
    """

    export_conda_env_to_file(file_name=file_name)

    # Add the updated environment file
    cmd = f"git add {file_name}"
    subprocess.Popen(cmd).communicate()

    print(f"{file_name}: Updated or unchanged")


def check_conda_env_update() -> bool:
    """Use after pull, merge or checkout.
        We want to check if the environment file has changed before updating the conda environment as this can take ~2mins

    Returns:
        bool: True if the conda environment requires updating
    """

    # The file that is up to date according to git
    new_env_file = "environment.yml"

    # The temp file exported from current conda env on your system
    current_env_file = "env_temp.yml"
    export_conda_env_to_file(file_name=current_env_file)

    # Compare the old and new files. True if files have same content, False if different
    files_identical = filecmp.cmp(new_env_file, current_env_file, shallow=False)

    # Tidy up - Remove the temporary file
    os.remove(current_env_file)

    # If the files are not identical then the conda environment needs updating
    env_requires_update = not files_identical

    return env_requires_update


def get_conda_env_name() -> str:
    """Get the conda environment name from the environment.yml file

    Returns:
        str: name of the conda environment
    """

    with open("environment.yml", "r") as file:
        name = file.readline().split("name: ")[1].split("\n")[0]

    return name


def update_conda_env():
    """Update the conda environment if the environment file has changed"""

    env_name = get_conda_env_name()

    if check_conda_env_update():

        print(f"Updating conda env {env_name} from environment.yml file")

        # Update the conda env
        cmd = f"conda env update -n {env_name} -f environment.yml"
        subprocess.Popen(cmd).communicate()

        print(f"conda env '{env_name}': Updated")
    else:
        print(f"conda env '{env_name}': Unchanged")
